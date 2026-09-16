"""Generate DAPE / RAM tag captions for a folder of LQ images.

ScaleResfusion conditions the FLUX.2 backbone on image tags produced by DAPE
(a RAM model fine-tuned for degraded inputs, proposed in SeeSR and used as in OSEDiff). This script writes
one caption file per image to ``{input_dir}/txt/{name}.txt`` -- exactly the layout
``test_resfusion_flux2.py`` expects.

Example
-------
    python generate_captions.py \
        --input_image path/to/test_LR \
        --ram_path    path/to/ram_swin_large_14m.pth \
        --ram_ft_path path/to/DAPE.pth

Weights
-------
    RAM : https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth
    DAPE: https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from PIL import Image
from torchvision import transforms

from my_utils.preprocess import list_images, prepare_input_image
from ram import inference_ram as inference
from ram.models.ram_lora import ram

tensor_transforms = transforms.Compose([transforms.ToTensor()])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DAPE captions for ScaleResfusion")
    parser.add_argument("--input_image", "-i", type=str, required=True,
                        help="directory of LQ images (or a single image file)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="where to write the caption txt files (default: {input_dir}/txt)")
    parser.add_argument("--ram_path", type=str, required=True,
                        help="RAM weights (ram_swin_large_14m.pth)")
    parser.add_argument("--ram_ft_path", type=str, default="",
                        help="DAPE LoRA weights (DAPE.pth). Leave empty to use vanilla RAM.")
    parser.add_argument("--prompt", type=str, default="",
                        help="extra user prompt appended after the tags")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = list_images(args.input_image)
    input_dir = args.input_image if os.path.isdir(args.input_image) else os.path.dirname(args.input_image)
    output_dir = args.output_dir or os.path.join(input_dir, "txt")
    os.makedirs(output_dir, exist_ok=True)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    # DAPE = RAM (swin-L) + LoRA fine-tuned on degraded images
    dape = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit="swin_l")
    dape.eval()
    dape.to(args.device, dtype=weight_dtype)

    print(f"There are {len(image_paths)} images.")
    for idx, image_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # tags are predicted on the upsampled LQ image, as in the evaluation protocol
        image = Image.open(image_path).convert("RGB")
        image, _, _ = prepare_input_image(image, upscale=args.upscale, process_size=args.process_size)

        with torch.no_grad():
            lq = tensor_transforms(image).unsqueeze(0).to(args.device)
            lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
            captions = inference(lq_ram, dape)
        # same formatting as the OSEDiff / ScaleResfusion evaluation scripts
        caption = f"{captions[0]}, {args.prompt},"

        with open(os.path.join(output_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"[{idx}/{len(image_paths)}] {image_path} | {caption}")

    print(f"Done. Captions saved to {output_dir}")


if __name__ == "__main__":
    main()
