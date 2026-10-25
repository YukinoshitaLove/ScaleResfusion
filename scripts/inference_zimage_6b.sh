#!/usr/bin/env bash
# ScaleResfusion (Z-Image 6B): caption -> restore -> evaluate one benchmark split.
#
# Usage (the same six positional arguments as inference_flux2_4b.sh):
#   bash scripts/inference_zimage_6b.sh <LR_DIR> <HR_DIR> <OUT_DIR> <LORA_CKPT> [RAM_PTH] [DAPE_PTH]
#
# Existing captions are read from LR_DIR/txt/{image_stem}.txt. If captions are
# missing, both RAM and DAPE weights are required for offline caption generation.
# That writes LR_DIR/txt; provide a writable input directory. No staging is done.
set -euo pipefail

LR_DIR=${1:?LR_DIR}
HR_DIR=${2:?HR_DIR}
OUT_DIR=${3:?OUT_DIR}
LORA_CKPT=${4:?LORA_CKPT}
RAM_PTH=${5:-}
DAPE_PTH=${6:-}

if [[ ! -d "${LR_DIR}" || ! -d "${HR_DIR}" ]]; then
  printf '%s\n' 'LR_DIR and HR_DIR must be existing directories.' >&2
  exit 1
fi
if [[ ! -f "${LORA_CKPT}" ]]; then
  printf '%s\n' 'LORA_CKPT must be an existing trained Z-Image generator checkpoint file.' >&2
  exit 1
fi
if [[ ( -n "${RAM_PTH}" && -z "${DAPE_PTH}" ) || ( -z "${RAM_PTH}" && -n "${DAPE_PTH}" ) ]]; then
  printf '%s\n' 'Provide both RAM_PTH and DAPE_PTH, or omit both and use existing captions.' >&2
  exit 1
fi

# 1) Generate captions only if at least one input image lacks its fixed txt file.
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
  if [[ -z "${RAM_PTH}" || -z "${DAPE_PTH}" ]]; then
    printf '%s\n' 'Missing LR_DIR/txt captions. Supply both RAM_PTH and DAPE_PTH, prepare captions offline, or use the Python CLI with --prompt.' >&2
    exit 1
  fi
  if [[ ! -f "${RAM_PTH}" || ! -f "${DAPE_PTH}" ]]; then
    printf '%s\n' 'RAM_PTH and DAPE_PTH must both point to existing files.' >&2
    exit 1
  fi
  python generate_captions.py \
      --input_image "${LR_DIR}" \
      --ram_path "${RAM_PTH}" \
      --ram_ft_path "${DAPE_PTH}"
fi

# 2) Restore (4 Euler sampling steps at --flowT 20 and --rsr 1.0).
# bf16 is the default; fp16 is not guaranteed stable.
python test_resfusion_zimage.py \
    --input_image "${LR_DIR}" \
    --output_dir "${OUT_DIR}" \
    --pretrained_model_name_or_path Tongyi-MAI/Z-Image \
    --resfusion_path "${LORA_CKPT}" \
    --rsr 1.0 --flowT 20 --upscale 4 --process_size 512 \
    --align_method adain --mixed_precision bf16 --seed 42

# 3) Metrics
python test_metrics.py \
    --inp_imgs "${OUT_DIR}" \
    --gt_imgs "${HR_DIR}" \
    --log "${OUT_DIR}"
