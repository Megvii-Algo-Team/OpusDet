#!/usr/bin/env bash
# Shared by playground/test_*.sh
#
# Primary recipe: OPUS DINOv3 ConvNeXt-B all @ 260k, four modes:
#   MODE="text_only visual.I.1 visual.G.16 text_visual.G.16"
#   BMK="coco lvis-mv odinw35"
#
# Expects before run_test:
#   PRODUCT     — config dir under configs/
#   CHECKPOINT  — .pth path
# Optional:
#   BMK         — coco | lvis | lvis-mv | odinw13 | odinw35
#   CONFIG_NAME — relative to configs/ without .py
#   MODE        — space-separated; each run sets model.mode via --cfg-options
#   PROMPT_PATH — visual.G pkl for coco/lvis; odinw35 uses per-subset pkl via VISUAL_PROMPT_CKPT_DIR
#   ROOT_WORK_DIR — if set, relative CHECKPOINT / VISUAL_PROMPT_WEIGHTS / PROMPT_PATH
#                   (and inferred work_dirs/… prompts) are prefixed with this root (e.g. /mnt/s3fs)
#   REGEN_VISUAL_PROMPTS — 1 to regen pkl before visual.G (default 0)
#   GEN_VISUAL_PROMPTS_ONLY — 1 to only regen visual prompts for BMK, skip eval
#   VISUAL_PROMPT_MODEL / VISUAL_PROMPT_WEIGHTS — for regen (default: PRODUCT o365 cfg + CHECKPOINT)
#   ROOT_DATA_DIR — dataset root (same layout as repo ``data/``: coco/, odinw/, …).
#                   Default ``data`` / ``$REPO_ROOT/data``. Exported for Config.fromfile.
#   IMAGE_SIZE      — speed_test only; ``;``-separated HxW or square side (e.g. 640 or 640;800)
#   DECODER_EARLY_EXIT_LAYER — speed_test / run_test; space-separated layer ids or none
#   SPEED_NUM_CLASSES — speed_test visual.I*: max classes per image (default 1)
#   GPUS / NNODES / NODE_RANK / PORT / MASTER_ADDR

is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
PORT=${PORT:-${MLP_WORKER_0_PORT:-29501}}
MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}

# Relative work_dirs paths: ${ROOT_WORK_DIR}/<path> when ROOT_WORK_DIR is set
_resolve_work_dir_path() {
  local p="$1"
  [[ -z "$p" ]] && return 0
  if [[ "$p" == /* ]]; then
    echo "$p"
    return 0
  fi
  if [[ -z "${ROOT_WORK_DIR:-}" ]]; then
    echo "$p"
    return 0
  fi
  p="${p#./}"
  echo "${ROOT_WORK_DIR%/}/${p}"
}

resolve_checkpoint_path() {
  _resolve_work_dir_path "$1"
}

checkpoint_exists() {
  local p
  p="$(resolve_checkpoint_path "$1")"
  [[ -n "$p" && -f "$p" ]]
}

_resolve_checkpoint_vars() {
  if [[ -n "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(resolve_checkpoint_path "$CHECKPOINT")"
  fi
  if [[ -n "${VISUAL_PROMPT_WEIGHTS:-}" ]]; then
    VISUAL_PROMPT_WEIGHTS="$(resolve_checkpoint_path "$VISUAL_PROMPT_WEIGHTS")"
  fi
  if [[ -n "${PROMPT_PATH:-}" ]]; then
    PROMPT_PATH="$(_resolve_work_dir_path "$PROMPT_PATH")"
  fi
}

_prompt_ckpt_subdir() {
  local weights="$1"
  local parent ckpt
  parent="$(basename "$(dirname "$weights")")"
  ckpt="$(basename "$weights" .pth)"
  echo "${parent}_${ckpt}"
}

# Config.fromfile / torchrun only see exported env. Shell assign without export
# (ROOT_DATA_DIR=/path) is invisible to python — export here before any launch.
# ROOT_WORK_DIR stays shell-only (path prefix in this script); no need to export.
_export_python_env() {
  [[ -n "${ROOT_DATA_DIR:-}" ]] && export ROOT_DATA_DIR
}

# Dataset root: ROOT_DATA_DIR if set, else repo ``data/``. Same tree for coco/lvis/odinw.
_data_root() {
  local r="${ROOT_DATA_DIR:-$REPO_ROOT/data}"
  echo "${r%/}"
}

# Map BMK -> config stem and default PROMPT_PATH (coco/lvis).
# Prints: "<config_name>|<prompt_path>" (does not mutate global CONFIG_NAME).
_resolve_bmk() {
  local bmk="$1"
  local p="${PRODUCT:?set PRODUCT}"
  local dirn pkl config_name prompt_path=""

  case "$bmk" in
    coco|o365)
      config_name="${p}/${p}_pretrain_o365"
      pkl="instances_train2017.pkl"
      ;;
    lvis)
      config_name="${p}/lvis/${p}_pretrain_zeroshot_lvis"
      pkl="lvis_v1_train_od.pkl"
      ;;
    lvis-mv|lvis_mv|mini-lvis)
      config_name="${p}/lvis/${p}_pretrain_zeroshot_mini-lvis"
      pkl="lvis_v1_train_od.pkl"
      ;;
    odinw13)
      config_name="${p}/odinw/${p}_pretrain_odinw13"
      pkl=""
      ;;
    odinw35)
      config_name="${p}/odinw/${p}_pretrain_odinw35"
      pkl=""
      ;;
    *)
      echo "test_common: unknown BMK='$bmk'" >&2
      return 1
      ;;
  esac

  if [[ -n "${PROMPT_PATH:-}" ]]; then
    prompt_path="$PROMPT_PATH"
  elif [[ -n "$pkl" && -n "${CHECKPOINT:-}" ]]; then
    dirn="$(_prompt_ckpt_subdir "$CHECKPOINT")"
    prompt_path="$(_resolve_work_dir_path "work_dirs/visual_prompts/${dirn}/preds/${pkl}")"
  fi

  echo "${config_name}|${prompt_path}"
}

# ODinW-35: per-subset pkl under work_dirs/visual_prompts/<ckpt_dir>/preds/<prefix>.pkl
# Anns + images colocated under ``$(_data_root)/odinw/`` (same as eval configs).
gen_odinw35_visual_prompts() {
  _export_python_env
  _resolve_checkpoint_vars

  local odinw_root
  odinw_root="$(_data_root)/odinw"
  echo "[gen_odinw35] odinw_root=$odinw_root"

  local vp_model="${VISUAL_PROMPT_MODEL:-configs/${PRODUCT}/${PRODUCT}_pretrain_o365.py}"
  local vp_weights="${VISUAL_PROMPT_WEIGHTS:-${CHECKPOINT:-}}"
  if [[ -z "$vp_weights" ]]; then
    echo "gen_odinw35: set VISUAL_PROMPT_WEIGHTS or CHECKPOINT" >&2
    return 1
  fi
  vp_weights="$(resolve_checkpoint_path "$vp_weights")"

  local sample_num=32 batch_sz=8
  local ckpt_dir preds_dir out_root
  ckpt_dir="$(_prompt_ckpt_subdir "$vp_weights")"
  out_root="$(_resolve_work_dir_path "work_dirs")"
  preds_dir="${out_root}/visual_prompts/${ckpt_dir}/preds"
  mkdir -p "$preds_dir"

  local _ODINW35_ENTRIES
  read -r -d '' _ODINW35_ENTRIES <<'EOF' || true
AerialMaritimeDrone_large|AerialMaritimeDrone/large/train/annotations_without_background.json|AerialMaritimeDrone/large/train/
AerialMaritimeDrone_tiled|AerialMaritimeDrone/tiled/train/annotations_without_background.json|AerialMaritimeDrone/tiled/train/
AmericanSignLanguageLetters|AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/annotations_without_background.json|AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/
Aquarium|Aquarium/Aquarium Combined.v2-raw-1024.coco/train/annotations_without_background.json|Aquarium/Aquarium Combined.v2-raw-1024.coco/train/
BCCD|BCCD/BCCD.v3-raw.coco/train/annotations_without_background.json|BCCD/BCCD.v3-raw.coco/train/
boggleBoards|boggleBoards/416x416AutoOrient/export/train_annotations_without_background.json|boggleBoards/416x416AutoOrient/export/
brackishUnderwater|brackishUnderwater/960x540/train/annotations_without_background.json|brackishUnderwater/960x540/train/
ChessPieces|ChessPieces/Chess Pieces.v23-raw.coco/train/annotations_without_background.json|ChessPieces/Chess Pieces.v23-raw.coco/train/
CottontailRabbits|CottontailRabbits/train/annotations_without_background.json|CottontailRabbits/train/
dice|dice/mediumColor/export/train_annotations_without_background.json|dice/mediumColor/export/
DroneControl|DroneControl/Drone Control.v3-raw.coco/train/annotations_without_background.json|DroneControl/Drone Control.v3-raw.coco/train/
EgoHands_generic|EgoHands/generic/train/annotations_without_background.json|EgoHands/generic/train/
EgoHands_specific|EgoHands/specific/train/annotations_without_background.json|EgoHands/specific/train/
HardHatWorkers|HardHatWorkers/raw/train/annotations_without_background.json|HardHatWorkers/raw/train/
MaskWearing|MaskWearing/raw/train/annotations_without_background.json|MaskWearing/raw/train/
MountainDewCommercial|MountainDewCommercial/train/annotations_without_background.json|MountainDewCommercial/train/
NorthAmericaMushrooms|NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/annotations_without_background.json|NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/
openPoetryVision|openPoetryVision/512x512/train/annotations_without_background.json|openPoetryVision/512x512/train/
OxfordPets_by_breed|OxfordPets/by-breed/train/annotations_without_background.json|OxfordPets/by-breed/train/
OxfordPets_by_species|OxfordPets/by-species/train/annotations_without_background.json|OxfordPets/by-species/train/
PKLot|PKLot/640/train/annotations_without_background.json|PKLot/640/train/
Packages|Packages/Raw/train/annotations_without_background.json|Packages/Raw/train/
PascalVOC|PascalVOC/train/annotations_without_background.json|PascalVOC/train/
pistols|pistols/export/train_annotations_without_background.json|pistols/export/
plantdoc|plantdoc/416x416/train/annotations_without_background.json|plantdoc/416x416/train/
pothole|pothole/train/annotations_without_background.json|pothole/train/
Raccoons|Raccoon/Raccoon.v2-raw.coco/train/annotations_without_background.json|Raccoon/Raccoon.v2-raw.coco/train/
selfdrivingCar|selfdrivingCar/fixedLarge/export/train_annotations_without_background.json|selfdrivingCar/fixedLarge/export/
ShellfishOpenImages|ShellfishOpenImages/raw/train/annotations_without_background.json|ShellfishOpenImages/raw/train/
ThermalCheetah|ThermalCheetah/train/annotations_without_background.json|ThermalCheetah/train/
thermalDogsAndPeople|thermalDogsAndPeople/train/annotations_without_background.json|thermalDogsAndPeople/train/
UnoCards|UnoCards/raw/train/annotations_without_background.json|UnoCards/raw/train/
VehiclesOpenImages|VehiclesOpenImages/416x416/train/annotations_without_background.json|VehiclesOpenImages/416x416/train/
WildfireSmoke|WildfireSmoke/train/annotations_without_background.json|WildfireSmoke/train/
websiteScreenshots|websiteScreenshots/train/annotations_without_background.json|websiteScreenshots/train/
EOF

  local prefix ann_rel img_rel ann_path img_prefix out_pkl ann_base gen_default_pkl
  local _ok=0 _skip=0 _fail=0

  while IFS='|' read -r prefix ann_rel img_rel; do
    [[ -z "$prefix" ]] && continue

    ann_path="${odinw_root}/${ann_rel}"
    img_prefix="${odinw_root}/${img_rel}"
    out_pkl="${preds_dir}/${prefix}.pkl"
    ann_base=$(basename "$ann_path" .json)
    gen_default_pkl="${preds_dir}/${ann_base}.pkl"

    if [[ ! -f "$ann_path" ]]; then
      echo "[SKIP] $prefix — missing ann: $ann_path" >&2
      _skip=$((_skip + 1))
      continue
    fi

    echo "=== $prefix ==="
    echo "  ann:  $ann_path"
    echo "  img:  $img_prefix"
    echo "  out:  $out_pkl"

    rm -f "$gen_default_pkl"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 tools/gen_visual_prompts.py \
      --input-json-path "$ann_path" \
      --data-prefix "$img_prefix" \
      --mode visual.I \
      --model "$vp_model" \
      --weights "$vp_weights" \
      --sample-num-limit "$sample_num" \
      --batch-size "$batch_sz" \
      --out-dir "$out_root"

    if [[ ! -f "$gen_default_pkl" ]]; then
      echo "[FAIL] $prefix — expected $gen_default_pkl after gen" >&2
      _fail=$((_fail + 1))
      continue
    fi
    mv -f "$gen_default_pkl" "$out_pkl"
    echo "[OK] $out_pkl"
    _ok=$((_ok + 1))
  done <<< "$_ODINW35_ENTRIES"

  echo "gen_odinw35 done: ok=$_ok skip=$_skip fail=$_fail -> $preds_dir"
  [[ "$_fail" -eq 0 ]]
}

_regen_visual_prompt_if_needed() {
  local mode="$1"
  local bmk="${2:-coco}"
  [[ "$mode" == *"visual.G"* ]] || return 0
  is_true "${REGEN_VISUAL_PROMPTS:-0}" || return 0

  local model="${VISUAL_PROMPT_MODEL:-configs/${PRODUCT}/${PRODUCT}_pretrain_o365.py}"
  local weights="${VISUAL_PROMPT_WEIGHTS:-$CHECKPOINT}"
  local root json data_prefix sample_num batch_sz
  root="$(_data_root)"

  case "$bmk" in
    lvis|mini-lvis|lvis-mv|lvis_mv)
      json="${root}/coco/annotations/lvis_v1_train_od.json"
      data_prefix="${root}/coco"
      sample_num=32
      batch_sz=16
      ;;
    coco|o365)
      json="${root}/coco/annotations/instances_train2017.json"
      data_prefix="${root}/coco/train2017"
      sample_num=32
      batch_sz=16
      ;;
    odinw35)
      echo "[test_common] REGEN_VISUAL_PROMPTS: ODinW-35 per-subset embeddings"
      VISUAL_PROMPT_WEIGHTS="$weights" \
      VISUAL_PROMPT_MODEL="$model" \
      gen_odinw35_visual_prompts
      return $?
      ;;
    *)
      echo "test_common: REGEN_VISUAL_PROMPTS=1 unsupported for BMK='$bmk' (coco|lvis|lvis-mv|odinw35)" >&2
      return 1
      ;;
  esac

  echo "[test_common] REGEN_VISUAL_PROMPTS: $json -> visual_prompts (mode visual.I)"
  local out_root
  out_root="$(_resolve_work_dir_path "work_dirs")"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" python3 tools/gen_visual_prompts.py \
    --input-json-path "$json" \
    --data-prefix "$data_prefix" \
    --sample-num-limit "$sample_num" \
    --batch-size "$batch_sz" \
    --mode visual.I \
    --model "$model" \
    --weights "$weights" \
    --out-dir "$out_root"
}

_run_one() {
  local mode="$1"
  local bmk="$2"
  local config_name="$3"
  local prompt_path="${4:-}"
  local _cfg="configs/${config_name}.py"
  local _ckpt="${CHECKPOINT:?set CHECKPOINT}"
  local _work_dir
  local _mode_tag="${mode//./_}"
  local cfg_opts="model.mode=${mode}"

  _export_python_env

  if [[ ! -f "$_cfg" ]]; then
    echo "test_common: config not found: $_cfg (skip)" >&2
    return 1
  fi
  if ! checkpoint_exists "$_ckpt"; then
    echo "test_common: checkpoint not found: $_ckpt (skip)" >&2
    return 1
  fi

  if [[ "$mode" == *"visual.G"* ]]; then
    if [[ "$bmk" == odinw35 || "$bmk" == odinw13 ]]; then
      export VISUAL_PROMPT_CKPT_DIR="$(_prompt_ckpt_subdir "$_ckpt")"
      echo "[test_common] odinw visual.G: VISUAL_PROMPT_CKPT_DIR=${VISUAL_PROMPT_CKPT_DIR}"
    elif [[ -z "${prompt_path:-}" ]]; then
      echo "test_common: mode=$mode needs PROMPT_PATH (or CHECKPOINT to infer)" >&2
      return 1
    else
      cfg_opts+=" model.test_cfg.prompt_path=${prompt_path}"
    fi
  fi

  if [[ "$config_name" == */lvis/* || "$config_name" == lvis/* ]]; then
    cfg_opts+=" env_cfg.dist_cfg.timeout=14400"
  fi
  if [[ "$bmk" == odinw* ]]; then
    cfg_opts+=" env_cfg.dist_cfg.timeout=7200"
  fi

  if [[ -n "${WORK_DIR:-}" ]]; then
    _work_dir="${WORK_DIR}_${_mode_tag}"
  else
    _work_dir="work_dirs/test/${PRODUCT}/${bmk}/${_mode_tag}"
  fi
  _work_dir="$(_resolve_work_dir_path "$_work_dir")"

  echo "[test] bmk=$bmk mode=$mode cfg=$_cfg ckpt=$_ckpt work_dir=$_work_dir"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
  torchrun \
      --nproc_per_node="$GPUS" \
      --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$PORT" \
      tools/test.py \
      "$_cfg" \
      "$_ckpt" \
      --launcher pytorch \
      --work-dir "$_work_dir" \
      --cfg-options $cfg_opts
}

# Parse IMAGE_SIZE for speed_test: ``;``-separated entries; empty/none → one default entry.
_parse_image_size_entries() {
  local -n _pis_out="${1:?output array name}"
  local raw="${2:-}"

  _pis_out=()
  if [[ -n "$raw" ]]; then
    raw="$(echo "$raw" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  fi
  if [[ -z "$raw" || "${raw,,}" == "none" ]]; then
    _pis_out=('')
    return
  fi

  local -a _pis_tmp=()
  IFS=';' read -r -a _pis_tmp <<< "$raw"
  local _seg
  for _seg in "${_pis_tmp[@]}"; do
    _seg="$(echo "$_seg" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "$_seg" ]] && _pis_out+=("$_seg")
  done
  if [[ ${#_pis_out[@]} -eq 0 ]]; then
    _pis_out=('')
  fi
}

_run_speed_one() {
  local config_name="$1"
  local mode="$2"
  local layer="$3"
  local img_size="$4"
  local config_py="configs/${config_name}.py"
  local work_dir

  if [[ ! -f "$config_py" ]]; then
    echo "test_common: config not found: $config_py (skip)" >&2
    return 1
  fi

  if [[ "$config_name" == */* ]]; then
    work_dir="work_dirs/flops_speed/${config_name//\//_}_${PRODUCT}"
  else
    work_dir="work_dirs/flops_speed/${config_name}_${PRODUCT}"
  fi
  work_dir="$(_resolve_work_dir_path "$work_dir")"
  mkdir -p "$work_dir"

  local layer_arg=""
  if [[ -n "$layer" && ! "${layer,,}" =~ ^(none|null)$ ]]; then
    layer_arg="--decoder-early-exit-layer $layer"
  fi

  local img_arg=""
  if [[ -n "$img_size" && ! "${img_size,,}" =~ ^(none|null)$ ]]; then
    img_arg="--img-size $img_size"
  fi

  local speed_nc=0
  local speed_nc_arg=""
  if [[ "$mode" == *visual.I* ]]; then
    speed_nc="${SPEED_NUM_CLASSES:-1}"
    if [[ "$speed_nc" =~ ^[0-9]+$ ]] && [[ "$speed_nc" -gt 0 ]]; then
      speed_nc_arg="--speed-num-classes $speed_nc"
    fi
  fi

  echo "[speed_test] $config_py | mode=$mode | layer=$layer | img=$img_size | speed_num_classes=${speed_nc}"

  _export_python_env
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 tools/analysis_tools/get_flops.py \
    "$config_py" \
    --show-table \
    --speed-test \
    --switch-to-deploy \
    --skip-flops \
    $img_arg \
    $layer_arg \
    $speed_nc_arg \
    --prompt-mode "$mode" \
    --out-file "$work_dir"
}

speed_test() {
  local b m l s
  local config
  local layers="${DECODER_EARLY_EXIT_LAYER:-none}"
  local -a image_sizes
  _parse_image_size_entries image_sizes "${IMAGE_SIZE:-}"

  : "${PRODUCT:?set PRODUCT}"

  if [[ -n "${CONFIG_NAME:-}" ]]; then
    local bmk_one="${BMK:-coco}"
    bmk_one="${bmk_one%% *}"
    config="${CONFIG_NAME}"
    for s in "${image_sizes[@]}"; do
      for m in ${MODE:-text_only}; do
        if [[ -n "$layers" && ! "${layers,,}" =~ ^(none|null)$ ]]; then
          for l in $layers; do
            _run_speed_one "$config" "$m" "$l" "$s"
          done
        else
          _run_speed_one "$config" "$m" "" "$s"
        fi
      done
    done
    return
  fi

  for b in ${BMK:-coco}; do
    local resolved
    resolved="$(_resolve_bmk "$b")" || return 1
    config="${resolved%%|*}"
    for s in "${image_sizes[@]}"; do
      for m in ${MODE:-text_only}; do
        if [[ -n "$layers" && ! "${layers,,}" =~ ^(none|null)$ ]]; then
          for l in $layers; do
            _run_speed_one "$config" "$m" "$l" "$s"
          done
        else
          _run_speed_one "$config" "$m" "" "$s"
        fi
      done
    done
  done
}

run_test() {
  local _bmk_list _bmk _mode
  local _user_cfg="${CONFIG_NAME:-}"
  local _resolved config_name prompt_path

  : "${PRODUCT:?set PRODUCT}"
  _export_python_env
  _resolve_checkpoint_vars
  MODE="${MODE:-text_only visual.I.1 visual.G.16 text_visual.G.16}"

  # Only generate visual prompts (no eval). Forces REGEN for each BMK once.
  if is_true "${GEN_VISUAL_PROMPTS_ONLY:-0}"; then
    export REGEN_VISUAL_PROMPTS=1
    _bmk_list="${BMK:-coco lvis-mv odinw35}"
    if [[ -n "$_user_cfg" ]]; then
      _bmk_list="${_bmk_list%% *}"
    fi
    echo "[test_common] GEN_VISUAL_PROMPTS_ONLY=1 BMK=$_bmk_list (skip eval)"
    for _bmk in $_bmk_list; do
      _regen_visual_prompt_if_needed "visual.G.16" "$_bmk" || return 1
    done
    return
  fi

  if [[ -n "$_user_cfg" ]]; then
    _bmk_list="${BMK:-coco}"
    _bmk_list="${_bmk_list%% *}"
    _resolved="$(_resolve_bmk "$_bmk_list")" || return 1
    prompt_path="${_resolved##*|}"
    config_name="$_user_cfg"
    for _mode in $MODE; do
      _regen_visual_prompt_if_needed "$_mode" "$_bmk_list" || return 1
      _run_one "$_mode" "$_bmk_list" "$config_name" "$prompt_path" || true
    done
    return
  fi

  _bmk_list="${BMK:-coco lvis-mv odinw35}"
  for _bmk in $_bmk_list; do
    _resolved="$(_resolve_bmk "$_bmk")" || continue
    config_name="${_resolved%%|*}"
    prompt_path="${_resolved##*|}"
    for _mode in $MODE; do
      _regen_visual_prompt_if_needed "$_mode" "$_bmk" || return 1
      _run_one "$_mode" "$_bmk" "$config_name" "$prompt_path" || true
    done
  done
}
