#!/usr/bin/env bash
# Train OPUS DINOv3 ConvNeXt-B (O365 + GoldG).
set -euo pipefail
source "$(dirname "$0")/train_common.sh"

PRODUCT=opus_dinov3_convnext-b
BASENAME=opus_dinov3_convnext-b_text_visual_pretrain_o365_goldg
RESUME="${RESUME:-0}"
USE_AMP="${USE_AMP:-1}"
run_train
