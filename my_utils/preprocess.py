"""Shared test-time image preprocessing for ScaleResfusion."""
import glob
import os

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(path):
    """Return a sorted list of image paths from a directory, or ``[path]`` for a single file."""
    if os.path.isdir(path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        return sorted(set(files))
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Input path does not exist: {path}")


def prepare_input_image(image, upscale=4, process_size=512):
    """Upsample the LQ image to the output resolution and make it a multiple of 16.

    This mirrors the OSEDiff / ScaleResfusion evaluation protocol:
      1. if the LQ image is smaller than ``process_size // upscale`` on its short side,
         enlarge it first (``resize_flag`` is set so the output can be resized back);
      2. PIL resize by ``upscale`` (default x4) with Pillow's default filter (BICUBIC for RGB);
         the optional step-1 enlargement uses the same default filter;
      3. floor the width and height to a multiple of 16 (required by the FLUX.2 VAE) and
         resize to that size with LANCZOS (no cropping).

    Returns:
        (image, (ori_width, ori_height), resize_flag)
    """
    ori_width, ori_height = image.size
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        image = image.resize((int(scale * ori_width), int(scale * ori_height)))
        resize_flag = True
    image = image.resize((image.size[0] * upscale, image.size[1] * upscale))

    new_width = image.width - image.width % 16
    new_height = image.height - image.height % 16
    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image, (ori_width, ori_height), resize_flag
