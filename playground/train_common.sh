#!/usr/bin/env bash
# Shared by playground/train_*.sh
#
# Expects before run_train:
#   PRODUCT   — config dir under configs/ (e.g. opus_dinov3_convnext-b)
#   BASENAME  — config stem without .py (e.g. opus_dinov3_convnext-b_text_visual_pretrain_o365_goldg)
# Optional:
#   REPO_ROOT     — repo root containing tools/train.py (auto-detect)
#   HF_ENV         — optional HF_ENV.sh (sets HF_HOME / offline)
#   ROOT_DATA_DIR  — dataset root (passed through to configs; same as gen_odinw35)
#   GPUS/NNODES/NODE_RANK/PORT/MASTER_ADDR
#   RESUME         — default 0
#   USE_AMP        — default 1
#   WORK_DIR       — default work_dirs/${BASENAME}_amp
#   PKILL_BEFORE   — default 0 (safer for open-source; research used 1)

is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# playground -> repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"

if [[ -n "${HF_ENV:-}" && -f "$HF_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$HF_ENV"
elif [[ -f "$_SCRIPT_DIR/HF_ENV.sh" ]]; then
  # shellcheck source=/dev/null
  MODEL_DIR="${MODEL_DIR:-$HOME/.cache}" source "$_SCRIPT_DIR/HF_ENV.sh"
fi

cd "$REPO_ROOT" || exit 1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

GPUS=${GPUS:-${MLP_WORKER_GPU:-8}}
NNODES=${NNODES:-${MLP_WORKER_NUM:-1}}
NODE_RANK=${NODE_RANK:-${MLP_ROLE_INDEX:-0}}
PORT=${PORT:-${MLP_WORKER_0_PORT:-29500}}
MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}

run_train() {
  local _product="${PRODUCT:?set PRODUCT (e.g. opus_dinov3_convnext-b)}"
  local _basename="${BASENAME:?set BASENAME}"
  local _resume="${RESUME:-0}"
  local _amp="${USE_AMP:-1}"
  local _pkill="${PKILL_BEFORE:-0}"
  local _work_dir
  local _cfg="configs/${_product}/${_basename}.py"

  # Config.fromfile reads os.environ; export shell-only ROOT_DATA_DIR assigns.
  [[ -n "${ROOT_DATA_DIR:-}" ]] && export ROOT_DATA_DIR

  if [[ ! -f "$_cfg" ]]; then
    echo "train_common: config not found: $_cfg" >&2
    exit 1
  fi

  if [[ -n "${WORK_DIR:-}" ]]; then
    _work_dir="$WORK_DIR"
  else
    if is_true "$_amp"; then
      _work_dir="work_dirs/${_basename}_amp"
    else
      _work_dir="work_dirs/${_basename}"
    fi
  fi

  if is_true "$_pkill"; then
    pkill -f "python3" || true
  fi

  local args="$_cfg --launcher pytorch --work-dir ${_work_dir}"
  if is_true "$_amp"; then
    args="$args --amp"
  fi
  if is_true "$_resume"; then
    args="$args --resume"
  fi

  echo "[train] $args"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
  MSATTN_FORCE_FP32="${MSATTN_FORCE_FP32:-True}" torchrun \
      --nproc_per_node="$GPUS" \
      --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$PORT" \
      tools/train.py \
      $args
}
