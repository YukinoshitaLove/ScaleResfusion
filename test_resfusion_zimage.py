"""ScaleResfusion inference with the Z-Image 6B backbone.

For an image ``{input_dir}/{stem}.png``, the fixed caption location is
``{input_dir}/txt/{stem}.txt``. ``--prompt`` is used only when that file is missing.
This entry point does not run RAM or generate captions online.
"""
import argparse
import math
import os


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _checkpoint_file(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"checkpoint must be an existing file, got: {value!r}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ScaleResfusion (Z-Image 6B) inference")
    parser.add_argument("--input_image", "-i", type=str, required=True,
                        help="directory of LQ images (or a single image file)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="directory to save the restored images")
    parser.add_argument("--prompt", type=str, default=None,
                        help="fallback prompt when {input_dir}/txt/{stem}.txt does not exist")
    parser.add_argument("--save_prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="also save each image's prompt to {output_dir}/txt/")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="Tongyi-MAI/Z-Image", help="Z-Image base model id or local path")
    parser.add_argument("--resfusion_path", type=_checkpoint_file, required=True,
                        help="trained ScaleResfusion Z-Image LoRA checkpoint file "
                             "(gen_hq_transformer_lora.safetensors)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--process_size", type=_positive_int, default=512)
    parser.add_argument("--upscale", type=_positive_int, default=4)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"],
                        default="adain", help="output color correction w.r.t. the LQ input")
    parser.add_argument("--rsr", type=_positive_float, default=1.0,
                        help="residual ratio gamma; sampling starts at t* = 1 / (1 + gamma)")
    parser.add_argument("--flowT", type=_positive_int, default=20,
                        help="nominal rectified-flow steps before truncation "
                             "(20 -> 4 sampling steps at rsr=1.0)")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "fp32"],
                        default="bf16", help="bf16 is the default; fp16 stability is not guaranteed")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Parse/validate the CLI before importing CUDA/model dependencies. In particular,
    # --help and invalid/missing checkpoints do not require loading torch or weights.
    import torch
    from accelerate.utils import set_seed
    from PIL import Image
    from torchvision import transforms

    from my_utils.captions import get_caption_path, load_caption
    from my_utils.image_io import restored_tensor_to_pil
    from my_utils.preprocess import list_images, prepare_input_image
    from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
    from resfusion_zimage import Resfusion_zimage_test

    set_seed(args.seed)
    image_paths = list_images(args.input_image)
    input_dir = args.input_image if os.path.isdir(args.input_image) else os.path.dirname(args.input_image)
    if not image_paths:
        raise ValueError(f"No input images found: {args.input_image}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        os.makedirs(os.path.join(args.output_dir, "txt"), exist_ok=True)

    model = Resfusion_zimage_test(args)
    print(f"Loaded ScaleResfusion ({args.pretrained_model_name_or_path}), "
          f"{model.model.num_sampling_steps} sampling steps, rsr={args.rsr}, "
          f"precision={args.mixed_precision}.")
    print(f"There are {len(image_paths)} images.")
    to_tensor = transforms.ToTensor()

    for idx, image_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        with Image.open(image_path) as image:
            input_image = image.convert("RGB")
        input_image, (ori_width, ori_height), resize_flag = prepare_input_image(
            input_image, upscale=args.upscale, process_size=args.process_size
        )

        caption = load_caption(input_dir, image_path, args.prompt)
        if args.save_prompts:
            with open(get_caption_path(args.output_dir, image_path), "w", encoding="utf-8") as f:
                f.write(caption)
        print(f"[{idx}/{len(image_paths)}] {image_path} | prompt: {caption}")

        with torch.no_grad():
            lq = to_tensor(input_image).unsqueeze(0).to("cuda")
            lq = lq * 2 - 1
            output_image = model(lq, prompt=caption)
            output_pil = restored_tensor_to_pil(output_image[0])
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
