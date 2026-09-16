import torch
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
def _encode_vae_image(vae: AutoencoderKLFlux2, image: torch.Tensor, generator: torch.Generator):
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")

    image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    image_latents = Flux2KleinPipeline._patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std

    return image_latents

def decode_vae_image(vae: AutoencoderKLFlux2, latents: torch.Tensor, latent_ids: torch.Tensor):
    latents = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = Flux2KleinPipeline._unpatchify_latents(latents)
    output_image = vae.decode(latents).sample.clamp(-1, 1)

    return output_image

# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_image_ids
def _prepare_image_ids(
    image_latents: torch.Tensor,  # (B, C, H, W)
    scale: int = 10,
):
    r"""
    Generates 4D position IDs (T, H, W, L) for image latent features.

    This function builds a coordinate tensor for each spatial position in the input
    latent feature map. The coordinates are defined over four dimensions:
    time, height, width, and layer. The generated position IDs are shared across
    all samples in the batch.

    Args:
        image_latents (torch.Tensor):
            Input image latent tensor of shape (B, C, H, W), where B is the batch size,
            C is the channel dimension, and H/W are the spatial dimensions.
        scale (int, optional):
            The fixed value used as the time coordinate T for all positions.
            Defaults to 10.

    Returns:
        torch.Tensor:
            A tensor of shape (B, H * W, 4), where each position is represented as
            (T, H, W, L).

    Coordinate Components:
        - T: Fixed time coordinate, set to `scale`
        - H: Row index in the latent feature map
        - W: Column index in the latent feature map
        - L: Layer index, fixed to 0
    """

    batch_size, _, height, width = image_latents.shape

    # create time offset for reference image
    t_coords = torch.tensor([scale])
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t_coords, h, w, l)

    # Expand to batch: (B, H*W, 4)
    image_latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return image_latent_ids

# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_image_latents
def prepare_image_latents(
    vae: AutoencoderKLFlux2,
    images: torch.Tensor,  # (B, C, H, W)
    device,
    dtype,
    generator: torch.Generator | None = None,
):
    images = images.to(device=device, dtype=dtype)
    image_latents = _encode_vae_image(vae=vae, image=images, generator=generator)  # (B, 128, 32, 32)

    image_latent_ids = _prepare_image_ids(image_latents)
    image_latent_ids = image_latent_ids.to(device)

    # Pack latent
    image_latents = Flux2KleinPipeline._pack_latents(image_latents)  # (B, 1024, 128)

    return image_latents, image_latent_ids

# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_latents
def prepare_latents_from_image_latents(
    image_latents: torch.Tensor,
    image_ids: torch.Tensor,
    device,
    dtype,
):
    if image_latents is not None:
        latents = torch.randn_like(image_latents).to(device=device, dtype=dtype)
    else:
        raise ValueError("image_latents can not be None !")

    unpacked_image_latents = Flux2KleinPipeline._unpack_latents_with_ids(image_latents, image_ids)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(unpacked_image_latents)
    latent_ids = latent_ids.to(device)

    return latents, latent_ids