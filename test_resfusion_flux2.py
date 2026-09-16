"""ScaleResfusion inference with the FLUX.2-klein-4B backbone.

Example
-------
    python test_resfusion_flux2.py \
        --input_image  path/to/test_LR \
        --output_dir   path/to/output \
        --resfusion_path path/to/gen_hq_transformer_lora.safetensors

Captions
--------
Each input image ``{input_dir}/{name}.png`` is paired with a caption file
``{input_dir}/txt/{name}.txt`` (DAPE tags, see ``generate_captions.py``). If a caption
file is missing and ``--prompt`` is given, that prompt is used instead.
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

from my_utils.preprocess import list_images, prepare_input_image
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from resfusion_flux2 import Resfusion_flux2_test

tensor_transforms = transforms.Compose([transforms.ToTensor()])


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


def parse_args():
    parser = argparse.ArgumentParser(description="ScaleResfusion (FLUX.2-klein-4B) inference")
    # I/O
    parser.add_argument("--input_image", "-i", type=str, required=True,
                        help="directory of LQ images (or a single image file)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="directory to save the restored images")
    parser.add_argument("--prompt", type=str, default=None,
                        help="fallback prompt used when {input_dir}/txt/{name}.txt does not exist")
    parser.add_argument("--save_prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="also save the prompt of every image to {output_dir}/txt/")
    # models
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="black-forest-labs/FLUX.2-klein-base-4B",
                        help="FLUX.2-klein model id or local path")
    parser.add_argument("--resfusion_path", type=str, required=True,
                        help="ScaleResfusion LoRA checkpoint (gen_hq_transformer_lora.safetensors)")
    # image setting
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"],
                        default="adain", help="color correction of the output w.r.t. the LQ input")
    # RRF sampling setting
    parser.add_argument("--rsr", type=float, default=1.0,
                        help="residual ratio gamma; the sampling starts at t* = 1 / (1 + gamma)")
    parser.add_argument("--flowT", type=int, default=20,
                        help="nominal rectified-flow steps before truncation "
                             "(20 -> 4 sampling steps, 10 -> 2, 6 -> 1)")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (only used when no checkpoint is loaded)")
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "fp32"],
                        default="fp16")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    image_paths = list_images(args.input_image)
    input_dir = args.input_image if os.path.isdir(args.input_image) else os.path.dirname(args.input_image)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        os.makedirs(os.path.join(args.output_dir, "txt"), exist_ok=True)

    # initialize the model
    model = Resfusion_flux2_test(args)
    print(f"Loaded ScaleResfusion ({args.pretrained_model_name_or_path}), "
          f"{model.model.num_sampling_steps} sampling steps, rsr={args.rsr}, "
          f"precision={args.mixed_precision}.")
    print(f"There are {len(image_paths)} images.")

    for idx, image_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # upsample the LQ image and make it a multiple of 16
        input_image = Image.open(image_path).convert("RGB")
        input_image, (ori_width, ori_height), resize_flag = prepare_input_image(
            input_image, upscale=args.upscale, process_size=args.process_size
        )

        # caption
        caption = load_caption(input_dir, image_path, args.prompt)
        if args.save_prompts:
            with open(os.path.join(args.output_dir, "txt", f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write(caption)
        print(f"[{idx}/{len(image_paths)}] {image_path} | prompt: {caption}")

        # restore
        with torch.no_grad():
            lq = tensor_transforms(input_image).unsqueeze(0).to("cuda")
            lq = lq * 2 - 1
            output_image = model(lq, prompt=caption)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

            if args.align_method == "adain":
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == "wavelet":
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))

        output_pil.save(os.path.join(args.output_dir, f"{stem}.png"))

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
