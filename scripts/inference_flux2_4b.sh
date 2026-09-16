#!/usr/bin/env bash
# ScaleResfusion (FLUX.2-klein-4B): caption -> restore -> evaluate on one benchmark split.
#
# Usage:
#   bash scripts/inference_flux2_4b.sh <LR_DIR> <HR_DIR> <OUT_DIR> <LORA_CKPT> [RAM_PTH] [DAPE_PTH]
#
# Example (DRealSR, StableSR test set layout):
#   bash scripts/inference_flux2_4b.sh \
#       datasets/DrealSRVal_crop128/test_LR \
#       datasets/DrealSRVal_crop128/test_HR \
#       results/DrealSR_flux2_4b \
#       weights/ScaleResfusion-FLUX2-4B/gen_hq_transformer_lora.safetensors \
#       weights/ram_swin_large_14m.pth weights/DAPE.pth
set -euo pipefail

LR_DIR=${1:?LR_DIR}
HR_DIR=${2:?HR_DIR}
OUT_DIR=${3:?OUT_DIR}
LORA_CKPT=${4:?LORA_CKPT}
RAM_PTH=${5:-}
DAPE_PTH=${6:-}

# 1) captions (skipped if ${LR_DIR}/txt already exists or no RAM weights are given)
if [[ -n "${RAM_PTH}" && ! -d "${LR_DIR}/txt" ]]; then
  python generate_captions.py \
      --input_image "${LR_DIR}" \
      --ram_path "${RAM_PTH}" \
      --ram_ft_path "${DAPE_PTH}"
fi

# 2) restoration (4 sampling steps: --flowT 20, --rsr 1.0)
python test_resfusion_flux2.py \
    --input_image "${LR_DIR}" \
    --output_dir "${OUT_DIR}" \
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
    --resfusion_path "${LORA_CKPT}" \
    --rsr 1.0 --flowT 20 --upscale 4 --process_size 512 \
    --align_method adain --mixed_precision fp16 --seed 42

# 3) metrics
python test_metrics.py \
    --inp_imgs "${OUT_DIR}" \
    --gt_imgs "${HR_DIR}" \
    --log "${OUT_DIR}"
