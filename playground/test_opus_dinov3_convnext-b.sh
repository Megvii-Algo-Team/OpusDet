#!/usr/bin/env bash
# Eval OPUS DINOv3 ConvNeXt-B.
#
#   bash playground/test_opus_dinov3_convnext-b.sh
#   ROOT_WORK_DIR=/mnt/s3fs bash ...
#   REGEN_VISUAL_PROMPTS=1 bash ...
#   GEN_VISUAL_PROMPTS_ONLY=1 BMK="coco lvis-mv odinw35" bash ...
set -euo pipefail
source "$(dirname "$0")/test_common.sh"

# ---------- speed_test (no CHECKPOINT; uses get_flops --speed-test) ----------
# PRODUCT=opus_dinov3_convnext-b
# BMK="coco"
# MODE="text_only visual.I.1"
# IMAGE_SIZE=640
# DECODER_EARLY_EXIT_LAYER="none"
# speed_test

# ---------- main · all @ 260k ----------
PRODUCT=opus_dinov3_convnext-b
CHECKPOINT="${CHECKPOINT:-work_dirs/${PRODUCT}_text_visual_pretrain_all_amp/opus_dinov3_convnext-b_text_visual_pretrain_all_amp_iter_260000-2c4dd683.pth}"
if ! checkpoint_exists "$CHECKPOINT"; then
  echo "missing CHECKPOINT=$(resolve_checkpoint_path "$CHECKPOINT")" >&2
  exit 1
fi
BMK="${BMK:-coco lvis-mv odinw35}"

# visual.I.1 — one visual prompt per class (image-level)
MODE="visual.I.1" run_test

# visual.G.16 — 16 visual prompts per class; regen pkl before eval if needed
MODE="visual.G.16" REGEN_VISUAL_PROMPTS=1 run_test

# text_only — CLIP text prompts only
MODE="text_only" run_test

# text_visual.G.16 — text + 16 visual prompts
MODE="text_visual.G.16" run_test

# ---------- main · o365_goldg (same recipe as train_opus_dinov3_convnext-b.sh) ----------
# PRODUCT=opus_dinov3_convnext-b
# BMK="coco lvis-mv"
# for iter in 100000; do
#   CHECKPOINT=work_dirs/${PRODUCT}_text_visual_pretrain_o365_goldg_amp/iter_${iter}.pth
#   if ! checkpoint_exists "$CHECKPOINT"; then
#     echo "[o365_goldg] skip missing ckpt: $(resolve_checkpoint_path "$CHECKPOINT")" >&2
#     continue
#   fi
#   # visual.I.1 — one visual prompt per class
#   MODE="visual.I.1" run_test
#   # visual.G.16 — regen visual prompt pkl
#   MODE="visual.G.16" REGEN_VISUAL_PROMPTS=1 run_test
#   # text_only
#   MODE="text_only" run_test
#   # text + visual.G.16
#   MODE="text_visual.G.16" run_test
# done
