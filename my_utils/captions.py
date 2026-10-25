"""Shared caption-file contract for ScaleResfusion inference."""
import os


def get_caption_path(input_dir, image_path):
    """Caption rule: ``{input_dir}/txt/{image_stem}.txt``."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(input_dir, "txt", f"{stem}.txt")


def load_caption(input_dir, image_path, fallback_prompt=None):
    txt_path = get_caption_path(input_dir, image_path)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if fallback_prompt is not None:
        return fallback_prompt
    raise FileNotFoundError(
        f"Caption file not found: {txt_path}\n"
        "Generate captions with `python generate_captions.py -i <input_dir> ...`, "
        "or pass --prompt to use a fixed prompt for images without a caption file."
    )
