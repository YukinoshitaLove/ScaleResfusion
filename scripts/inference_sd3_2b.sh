#!/usr/bin/env bash
# ScaleResfusion (SD3 Medium 2B): caption -> restore -> evaluate one supplied split.
# Usage:
#   bash scripts/inference_sd3_2b.sh <LR_DIR> <HR_DIR> <OUT_DIR> <LORA_CKPT> [RAM_PTH] [DAPE_PTH]
#
# Public caption layout is <LR_DIR>/txt/<image_stem>.txt, as in the 4B CLI.
# Missing captions require both RAM and DAPE weights and a writable LR_DIR/txt.
set -euo pipefail

LR_DIR=${1:?LR_DIR}
HR_DIR=${2:?HR_DIR}
OUT_DIR=${3:?OUT_DIR}
LORA_CKPT=${4:?LORA_CKPT}
RAM_PTH=${5:-}
DAPE_PTH=${6:-}

if [[ ! -d "${LR_DIR}" || ! -d "${HR_DIR}" ]]; then
  printf '%s\n' 'LR_DIR and HR_DIR must be existing image directories.' >&2
  exit 1
fi
if [[ ! -f "${LORA_CKPT}" ]]; then
  printf 'SD3 generator LoRA checkpoint not found: %s\n' "${LORA_CKPT}" >&2
  exit 1
fi

shopt -s nullglob
NEEDS_CAPTIONS=false
IMAGE_COUNT=0
for IMAGE in "${LR_DIR}"/*.{png,jpg,jpeg,bmp,webp,PNG,JPG,JPEG,BMP,WEBP}; do
  [[ -f "${IMAGE}" ]] || continue
  IMAGE_COUNT=$((IMAGE_COUNT + 1))
  STEM=${IMAGE##*/}
  STEM=${STEM%.*}
  if [[ ! -f "${LR_DIR}/txt/${STEM}.txt" ]]; then
    NEEDS_CAPTIONS=true
  fi
done
if [[ "${IMAGE_COUNT}" -eq 0 ]]; then
  printf '%s\n' 'No supported images found in LR_DIR.' >&2
  exit 1
fi
if [[ "${NEEDS_CAPTIONS}" == true ]]; then
  if [[ ! -f "${RAM_PTH}" || ! -f "${DAPE_PTH}" ]]; then
    printf '%s\n' \
        'Missing captions: supply LR_DIR/txt captions or both RAM_PTH and DAPE_PTH.' \
        'For a fixed prompt, run test_resfusion_sd3.py directly with --prompt.' >&2
    exit 1
  fi
  python generate_captions.py \
      --input_image "${LR_DIR}" \
      --ram_path "${RAM_PTH}" \
      --ram_ft_path "${DAPE_PTH}" \
      --upscale 4 --process_size 512 --mixed_precision fp16
fi

# 2) restoration (4 sampling steps at --flowT 20 --rsr 1.0).
python test_resfusion_sd3.py \
    --input_image "${LR_DIR}" \
    --output_dir "${OUT_DIR}" \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --resfusion_path "${LORA_CKPT}" \
    --lq_layers 23 --rsr 1.0 --flowT 20 --upscale 4 --process_size 512 \
    --align_method adain --mixed_precision fp16 --seed 42

# 3) metrics
python test_metrics.py \
    --inp_imgs "${OUT_DIR}" \
    --gt_imgs "${HR_DIR}" \
    --log "${OUT_DIR}"
