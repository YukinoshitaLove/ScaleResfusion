#!/usr/bin/env bash
# ScaleResfusion (FLUX.2-klein-9B): caption -> restore -> evaluate.
# Usage: bash scripts/inference_flux2_9b.sh <LR_DIR> <HR_DIR> <OUT_DIR> <LORA_CKPT> [RAM_PTH] [DAPE_PTH]
set -euo pipefail

LR_DIR=${1:?LR_DIR}
HR_DIR=${2:?HR_DIR}
OUT_DIR=${3:?OUT_DIR}
LORA_CKPT=${4:?LORA_CKPT}
RAM_PTH=${5:-}
DAPE_PTH=${6:-}

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
    printf '%s\n' 'Captions are missing: provide both RAM_PTH and DAPE_PTH, or generate LR_DIR/txt first.' >&2
    exit 2
  fi
  python generate_captions.py \
      --input_image "${LR_DIR}" \
      --ram_path "${RAM_PTH}" \
      --ram_ft_path "${DAPE_PTH}"
fi

python test_resfusion_flux2.py \
    --input_image "${LR_DIR}" \
    --output_dir "${OUT_DIR}" \
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-9B \
    --resfusion_path "${LORA_CKPT}" \
    --rsr 1.0 --flowT 20 --upscale 4 --process_size 512 \
    --align_method adain --mixed_precision fp16 --seed 42 --cpu_offload

python test_metrics.py \
    --inp_imgs "${OUT_DIR}" \
    --gt_imgs "${HR_DIR}" \
    --log "${OUT_DIR}"
