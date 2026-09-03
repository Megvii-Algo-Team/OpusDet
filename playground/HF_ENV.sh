#!/bin/bash
# Optional HuggingFace / Torch hub cache for OPUS ConvNeXt / CLIP.
# Usage: MODEL_DIR=/path/to/cache source playground/HF_ENV.sh
MODEL_DIR="${MODEL_DIR:-$HOME/.cache}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_DIR/torch}"
export HF_HOME="${HF_HOME:-$MODEL_DIR/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"

export HF_HUB_OFFLINE="${HF_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${HF_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_OFFLINE:-0}"

mkdir -p "$TORCH_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE"
echo "[OPUS] HF_HOME=$HF_HOME OFFLINE=$HF_HUB_OFFLINE"
