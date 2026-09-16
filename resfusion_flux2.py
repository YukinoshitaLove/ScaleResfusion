"""ScaleResfusion generator on the FLUX.2-klein backbone (inference-only release).

The generator predicts the *residual vector field* of Residual Rectified Flow (RRF):
sampling starts from a noisy low-quality (LQ) latent at the acceleration point
t* = 1 / (1 + gamma) and integrates the learned ODE back to the clean latent with a
handful of Euler steps. Only LoRA adapters on the frozen FLUX.2 transformer are trained;
the VAE and text encoder stay frozen.
"""
import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from accelerate import Accelerator, AutocastKwargs
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from peft import LoraConfig

from scheduler import ResfusionFlowMatchScheduler
from stableresfusion_utils.flux2 import (
    decode_vae_image,
    prepare_image_latents,
    prepare_latents_from_image_latents,
)


def get_flux2_lora_target_modules(transformer):
    """Return the names of the FLUX.2 linear layers that carry LoRA adapters.

    The released checkpoints were trained with exactly this target set, so it must stay
    in sync with them.
    """
    target_modules = []

    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 1) double-stream blocks (transformer_blocks)
        if (
            name.endswith(".attn.to_q")
            or name.endswith(".attn.to_k")
            or name.endswith(".attn.to_v")
            or name.endswith(".attn.to_out.0")
            or name.endswith(".attn.add_q_proj")
            or name.endswith(".attn.add_k_proj")
            or name.endswith(".attn.add_v_proj")
            or name.endswith(".attn.to_add_out")
            or name.endswith(".ff.linear_in")
            or name.endswith(".ff.linear_out")
            or name.endswith(".ff_context.linear_in")
            or name.endswith(".ff_context.linear_out")
        ):
            target_modules.append(name)
            continue

        # 2) single-stream blocks (single_transformer_blocks)
        if "single_transformer_blocks" in name and (
            name.endswith(".attn.to_qkv_mlp_proj") or name.endswith(".attn.to_out")
        ):
            target_modules.append(name)
            continue

        # 3) output projection
        if name == "proj_out":
            target_modules.append(name)
            continue

    return sorted(set(target_modules))


def initialize_transformer(pretrained_model_name_or_path, lora_rank=32, ckpt_file=None):
    """Load the frozen FLUX.2 transformer and attach the ScaleResfusion LoRA adapter.

    Args:
        pretrained_model_name_or_path: HF repo id or local path of FLUX.2-klein.
        lora_rank: LoRA rank used when no checkpoint is given (fresh adapter).
        ckpt_file: path to ``gen_hq_transformer_lora.safetensors``. When provided, the
            trained adapter is loaded and ``lora_rank`` is ignored.
    """
    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer"
    )
    transformer.requires_grad_(False)

    if ckpt_file is not None:
        transformer.load_lora_adapter(ckpt_file, adapter_name="default_transformer")
    else:
        target_modules = get_flux2_lora_target_modules(transformer)
        lora_conf = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_conf, adapter_name="default_transformer")

    return transformer


class Resfusion_flux2_gen(nn.Module):
    """Residual Rectified Flow generator: frozen FLUX.2 VAE + LoRA-adapted FLUX.2 transformer."""

    def __init__(self, args, ckpt_file=None):
        super().__init__()
        self.args = args

        # RRF sampling schedule. ``flowT`` is the nominal number of rectified-flow steps
        # over t in [0, 1]; the schedule is then truncated at the acceleration point
        # t* = 1 / (1 + rsr), so the number of transformer evaluations is smaller
        # (flowT=20 -> 4 steps, flowT=10 -> 2 steps, flowT=6 -> 1 step).
        self.noise_scheduler = ResfusionFlowMatchScheduler(
            num_train_timesteps=1000, shift=3.0, rsr=args.rsr
        )
        self.noise_scheduler.set_timesteps(num_inference_steps=args.flowT, device="cuda")
        self.noise_scheduler.sigmas = self.noise_scheduler.sigmas.cuda()

        # frozen FLUX.2 VAE
        self.vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        self.vae.requires_grad_(False)

        # FLUX.2 transformer carrying the ScaleResfusion LoRA adapter
        self.hq_transformer = initialize_transformer(
            args.pretrained_model_name_or_path, args.lora_rank, ckpt_file
        )
        self.lora_rank_transformer = args.lora_rank

    @property
    def num_sampling_steps(self):
        """Actual number of transformer evaluations per image."""
        return len(self.noise_scheduler.timesteps)

    @torch.no_grad()
    def resfusion_inference_loop(self, lq_image, scheduler, prompt_embeds, txt_ids, accelerator):
        """Run RRF sampling from the noisy LQ latent to the restored image.

        Args:
            lq_image: LQ image tensor in [-1, 1], shape (B, 3, H, W), already upsampled to
                the target resolution (H and W multiples of 16).
            scheduler: a ``ResfusionFlowMatchScheduler`` with timesteps already set.
            prompt_embeds / txt_ids: FLUX.2 text-encoder outputs for the caption.
            accelerator: ``accelerate.Accelerator`` used for autocast.

        Returns:
            Restored image tensor in [-1, 1], shape (B, 3, H, W).
        """
        # FLUX.2 VAE encoder (with batch-norm latent normalisation)
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            lq_encoded_control, lq_encoded_control_ids = prepare_image_latents(
                vae=self.vae, images=lq_image, device=accelerator.device, dtype=self.vae.dtype
            )

        # Initialise at the acceleration point:
        #   x_{t*} = gamma / (1 + gamma) * z_LQ + 1 / (1 + gamma) * eps
        noise, noise_ids = prepare_latents_from_image_latents(
            image_latents=lq_encoded_control,
            image_ids=lq_encoded_control_ids,
            device=accelerator.device,
            dtype=self.vae.dtype,
        )
        accelerate_point = 1.0 / (1.0 + scheduler.rsr)
        latent = (1.0 - accelerate_point) * lq_encoded_control + accelerate_point * noise

        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latent.shape[0]).to(device=latent.device)
            sigma = scheduler.sigmas[i]
            sigma_next = scheduler.sigmas[i + 1]
            with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
                # LQ latents are concatenated along the token dimension as conditioning
                latent_model_input = torch.cat([latent, lq_encoded_control], dim=1).to(self.vae.dtype)
                latent_image_ids = torch.cat([noise_ids, lq_encoded_control_ids], dim=1)
                # predict the residual vector field resv = gamma * R + eps - z_0
                with self.hq_transformer.cache_context("cond"):
                    resv_pred = self.hq_transformer(
                        hidden_states=latent_model_input,  # (B, image_seq_len, C)
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=txt_ids,  # (B, text_seq_len, 4)
                        img_ids=latent_image_ids,  # (B, image_seq_len, 4)
                        return_dict=False,
                    )[0]
                resv_pred = resv_pred[:, : latent.size(1)]
            # Euler step of the RRF ODE
            latent = latent + (sigma_next - sigma) * resv_pred

        # FLUX.2 VAE decoder
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
            output_image = decode_vae_image(vae=self.vae, latents=latent, latent_ids=noise_ids)

        return output_image


class Resfusion_flux2_test(nn.Module):
    """Inference wrapper: text encoding + RRF sampling for a single LQ image."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        if not torch.cuda.is_available():
            raise EnvironmentError("Resfusion_flux2_test requires CUDA.")
        if not args.resfusion_path or not os.path.isfile(args.resfusion_path):
            raise FileNotFoundError(
                "--resfusion_path must point to the ScaleResfusion LoRA checkpoint "
                f"(gen_hq_transformer_lora.safetensors), got: {args.resfusion_path!r}"
            )

        self.device = torch.device("cuda")
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.mixed_precision]

        # generator: frozen VAE + LoRA-adapted transformer
        self.model = Resfusion_flux2_gen(args, ckpt_file=args.resfusion_path)
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

        # text encoder (we only keep the tokenizer / text encoder of the pipeline)
        self.pipe = Flux2KleinPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
        )
        del self.pipe.transformer, self.pipe.vae, self.pipe.scheduler
        self.pipe.text_encoder.eval()
        self.pipe.to(self.device, dtype=self.weight_dtype)

        # autocast helper
        mixed_precision = "no" if args.mixed_precision == "fp32" else args.mixed_precision
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        prompt_embeds, txt_ids = self.pipe.encode_prompt(prompt=prompt, device=self.device)
        prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype)
        txt_ids = txt_ids.to(dtype=self.weight_dtype)
        return prompt_embeds, txt_ids

    @torch.no_grad()
    def forward(self, lq, prompt):
        prompt_embeds, txt_ids = self.encode_prompt(prompt)
        lq = lq.to(device=self.device, dtype=self.weight_dtype)
        return self.model.resfusion_inference_loop(
            lq_image=lq,
            scheduler=self.model.noise_scheduler,
            prompt_embeds=prompt_embeds,
            txt_ids=txt_ids,
            accelerator=self.accelerator,
        )
