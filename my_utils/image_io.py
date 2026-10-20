"""Convert restored tensors with the release's existing PNG quantisation order."""
import torch
from torchvision.transforms import ToPILImage


def restored_tensor_to_pil(tensor):
    image = tensor.cpu() * 0.5 + 0.5
    if not torch.isfinite(image).all():
        raise ValueError("The restoration contains non-finite pixel values.")
    # NumPy/Pillow cannot represent bfloat16; retain its affine rounding before casting.
    if image.dtype == torch.bfloat16:
        image = image.float()
    return ToPILImage()(image)
