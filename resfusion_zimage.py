"""ScaleResfusion inference on the Z-Image backbone.

This release loads a trained generator LoRA checkpoint and exposes only inference.
"""
import os
import warnings

import torch
from accelerate import Accelerator, AutocastKwargs
from diffusers import AutoencoderKL, ZImagePipeline

from ReferenceNet_Modules.zimage_referencenet import ZImageReferenceTransformer2DModel
from scheduler import ResfusionFlowMatchScheduler


def _require_checkpoint_file(ckpt_file):
    if not ckpt_file or not os.path.isfile(ckpt_file):
        raise FileNotFoundError(
            "A trained ScaleResfusion Z-Image LoRA checkpoint is required; "
            "--resfusion_path must point to an existing file "
            f"(gen_hq_transformer_lora.safetensors), got: {ckpt_file!r}"
        )
    return ckpt_file


def initialize_transformer(pretrained_model_name_or_path, ckpt_file):
    """Load the reference transformer and the required trained generator adapter.

    Adapter-loading errors intentionally propagate: inference must never fall back
    to a randomly initialised adapter or silently run the unadapted base model.
    """
    _require_checkpoint_file(ckpt_file)
    transformer = ZImageReferenceTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer"
    )
    transformer.requires_grad_(False)
    transformer.load_lora_adapter(
        ckpt_file, weight_name=os.path.basename(ckpt_file), adapter_name="default_transformer"
    )
    return transformer


class Resfusion_zimage_gen(torch.nn.Module):
    """Frozen Z-Image VAE and LoRA-adapted reference transformer for RRF sampling."""

    def __init__(self, args, ckpt_file):
        super().__init__()
        _require_checkpoint_file(ckpt_file)
        self.args = args
        self.noise_scheduler = ResfusionFlowMatchScheduler(
            num_train_timesteps=1000, shift=3.0, rsr=args.rsr
        )
        self.noise_scheduler.set_timesteps(num_inference_steps=args.flowT, device="cuda")
        self.noise_scheduler.sigmas = self.noise_scheduler.sigmas.cuda()

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.hq_transformer = initialize_transformer(args.pretrained_model_name_or_path, ckpt_file)

    @property
    def num_sampling_steps(self):
        """Number of Euler steps; each step makes one reference write and one read call."""
        return len(self.noise_scheduler.timesteps)

    @torch.no_grad()
    def resfusion_inference_loop(self, lq_image, scheduler, prompt_embeds, accelerator):
        """Restore an upsampled LQ image in [-1, 1], retaining the trained sampling path."""
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            # The VAE posterior is sampled, not replaced by its mean/mode.
            lq_encoded_control = self.vae.encode(lq_image).latent_dist.sample()
            lq_encoded_control = (lq_encoded_control - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        noise = torch.randn_like(lq_encoded_control)
        accelerate_point = 1.0 / (1.0 + scheduler.rsr)
        latent = (1.0 - accelerate_point) * lq_encoded_control + accelerate_point * noise
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latent.shape[0]).to(device=latent.device)
            sigma = scheduler.sigmas[i]
            sigma_next = scheduler.sigmas[i + 1]
            with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
                # LoRA is disabled only for this write call. The model's
                # apply_lora_scale decorator restores it before the read call.
                block_k, block_v, block_masks = self.hq_transformer(
                    x=lq_encoded_control.unsqueeze(2),
                    t=(1000 - timestep) / 1000,
                    cap_feats=prompt_embeds,
                    joint_attention_kwargs={
                        "reference_mode": "write",
                        "scale": 0.0,
                    },
                    return_dict=False,
                )
                resv_pred_list = self.hq_transformer(
                    x=latent.unsqueeze(2),
                    t=(1000 - timestep) / 1000,
                    cap_feats=prompt_embeds,
                    joint_attention_kwargs={
                        "reference_mode": "read",
                        "block_k_samples": block_k,
                        "block_v_samples": block_v,
                        "block_reference_attention_masks": block_masks,
                    },
                    return_dict=False,
                )[0]
                resv_pred = torch.stack([out.float() for out in resv_pred_list], dim=0)
                resv_pred = resv_pred.squeeze(2)
                resv_pred = -resv_pred
            latent = latent + (sigma_next - sigma) * resv_pred

        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            encoded_output = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            output_image = self.vae.decode(encoded_output).sample.clamp(-1, 1)
        return output_image


class Resfusion_zimage_test(torch.nn.Module):
    """Single-image text wrapper and CUDA inference entry point (bf16 by default)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        _require_checkpoint_file(args.resfusion_path)
        if not torch.cuda.is_available():
            raise EnvironmentError("Resfusion_zimage_test requires CUDA.")

        self.device = torch.device("cuda")
        precision = getattr(args, "mixed_precision", "bf16")
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision]
        if precision == "fp16":
            warnings.warn(
                "Z-Image fp16 inference is not guaranteed to be stable and may produce NaNs; "
                "use bf16 (default) when supported.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.model = Resfusion_zimage_gen(args, ckpt_file=args.resfusion_path)
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

        # Keep only the text components; never load a second transformer or VAE.
        self.pipe = ZImagePipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype,
        )
        self.pipe.text_encoder.eval()
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.to(self.device, dtype=self.weight_dtype)

        mixed_precision = "no" if precision == "fp32" else precision
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=[prompt],
            device=self.device,
            do_classifier_free_guidance=False,
        )
        if isinstance(prompt_embeds, (list, tuple)):
            prompt_embeds = [embed.to(dtype=self.weight_dtype) for embed in prompt_embeds]
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt):
        prompt_embeds = self.encode_prompt(prompt)
        lq = lq.to(device=self.device, dtype=self.weight_dtype)
        return self.model.resfusion_inference_loop(
            lq_image=lq,
            scheduler=self.model.noise_scheduler,
            prompt_embeds=prompt_embeds,
            accelerator=self.accelerator,
        )
