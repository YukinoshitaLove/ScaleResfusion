"""ScaleResfusion on Stable Diffusion 3 Medium (inference-only).

The frozen, independent LQ ReferenceNet writes image K/V at every Euler step;
the LoRA-adapted HQ transformer reads those features while predicting the
residual vector field. A trained generator checkpoint is always required.
"""

import os

import torch
from accelerate import Accelerator, AutocastKwargs
from diffusers import AutoencoderKL, StableDiffusion3Pipeline

from ReferenceNet_Modules.sd3_referencenet import (
    SD3ReferenceNetReadModel,
    SD3ReferenceNetWriteModel,
)
from Reference_Atten.sd3_reference_attn_processor import (
    set_sd3_reference_read_attn,
    set_sd3_reference_write_attn,
)
from scheduler import ResfusionFlowMatchScheduler


def _require_checkpoint(ckpt_file):
    if not ckpt_file or not os.path.isfile(ckpt_file):
        raise FileNotFoundError(
            "--resfusion_path must point to the ScaleResfusion SD3 generator LoRA "
            f"checkpoint (gen_hq_transformer_lora.safetensors), got: {ckpt_file!r}"
        )


def initialize_read_transformer(pretrained_model_name_or_path, ckpt_file):
    """Load the trained HQ adapter, inferring its rank and targets from the payload.

    Do not filter targets to Linear layers: SD3 checkpoints also contain the
    Conv2d LoRA on ``pos_embed.proj``. There is no fresh-adapter inference fallback.
    """
    _require_checkpoint(ckpt_file)
    transformer = SD3ReferenceNetReadModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer"
    )
    transformer = set_sd3_reference_read_attn(transformer)
    transformer.load_lora_adapter(
        ckpt_file, weight_name=os.path.basename(ckpt_file), adapter_name="default_transformer"
    )
    transformer.requires_grad_(False)
    return transformer


class Resfusion_sd3_gen(torch.nn.Module):
    """Frozen SD3 VAE + independent LQ writer + trained HQ reader."""

    def __init__(self, args, ckpt_file):
        super().__init__()
        _require_checkpoint(ckpt_file)
        if not torch.cuda.is_available():
            raise EnvironmentError("Resfusion_sd3_gen requires CUDA.")
        self.args = args

        self.noise_scheduler = ResfusionFlowMatchScheduler(
            num_train_timesteps=1000, shift=3.0, rsr=args.rsr
        )
        self.noise_scheduler.set_timesteps(num_inference_steps=args.flowT, device="cuda")
        self.noise_scheduler.sigmas = self.noise_scheduler.sigmas.cuda()

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.lq_transformer = SD3ReferenceNetWriteModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            num_layers=getattr(args, "lq_layers", 23),
        )
        self.lq_transformer = set_sd3_reference_write_attn(self.lq_transformer)
        self.lq_transformer.requires_grad_(False)
        self.hq_transformer = initialize_read_transformer(args.pretrained_model_name_or_path, ckpt_file)

    @property
    def num_sampling_steps(self):
        """Actual number of Euler steps (each runs the LQ writer and HQ reader)."""
        return len(self.noise_scheduler.timesteps)

    @torch.no_grad()
    def resfusion_inference_loop(
        self, lq_image, scheduler, prompt_embeds, pooled_prompt_embeds, accelerator
    ):
        """Run the original SD3 RRF path, with timesteps on the SD3 0..1000 scale."""
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            # SD3 uses a sampled posterior, not posterior.mode().
            lq_encoded_control = self.vae.encode(lq_image).latent_dist.sample()
            lq_encoded_control = (
                lq_encoded_control - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

        noise = torch.randn_like(lq_encoded_control)
        accelerate_point = 1.0 / (1.0 + scheduler.rsr)
        latent = (1.0 - accelerate_point) * lq_encoded_control + accelerate_point * noise
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latent.shape[0]).to(device=latent.device)
            sigma = scheduler.sigmas[i]
            sigma_next = scheduler.sigmas[i + 1]
            with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
                # Recompute the timestep-conditioned LQ K/V on every step.
                lq_block_k_samples, lq_block_v_samples = self.lq_transformer(
                    referencenet_cond=lq_encoded_control,
                    timestep=timestep,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                )
                resv_pred = self.hq_transformer(
                    hidden_states=latent,
                    block_k_samples=lq_block_k_samples,
                    block_v_samples=lq_block_v_samples,
                    timestep=timestep,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            latent = latent + (sigma_next - sigma) * resv_pred

        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            encoded_output = latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
            output_image = self.vae.decode(encoded_output).sample.clamp(-1, 1)
        return output_image


class Resfusion_sd3_test(torch.nn.Module):
    """CUDA inference wrapper, retaining both CLIP encoders and the T5 encoder."""

    def __init__(self, args):
        super().__init__()
        _require_checkpoint(args.resfusion_path)
        if not torch.cuda.is_available():
            raise EnvironmentError("Resfusion_sd3_test requires CUDA.")
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.mixed_precision]

        self.model = Resfusion_sd3_gen(args, ckpt_file=args.resfusion_path)
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

        # Load only text components; never allocate a second transformer or VAE.
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            scheduler=None,
            torch_dtype=self.weight_dtype,
        )
        self.pipe.text_encoder.eval()
        self.pipe.text_encoder_2.eval()
        self.pipe.text_encoder_3.eval()
        self.pipe.to(self.device, dtype=self.weight_dtype)

        mixed_precision = "no" if args.mixed_precision == "fp32" else args.mixed_precision
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=self.device,
            do_classifier_free_guidance=False,
        )
        return (
            prompt_embeds.to(dtype=self.weight_dtype),
            pooled_prompt_embeds.to(dtype=self.weight_dtype),
        )

    @torch.no_grad()
    def forward(self, lq, prompt):
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
        lq = lq.to(device=self.device, dtype=self.weight_dtype)
        return self.model.resfusion_inference_loop(
            lq_image=lq,
            scheduler=self.model.noise_scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            accelerator=self.accelerator,
        )
