import ast
import argparse
import json
import os
import re
import subprocess
import sys

# Repo root so ``opus`` imports work without setting PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock
from argparse import ArgumentParser
from tqdm import tqdm
from mmengine.logging import print_log
import mmengine
from mmcv.ops import nms
import numpy as np
from collections import defaultdict
import traceback
from opus.apis import PromptDetInferencer
from mmdet.evaluation import get_classes
from mmengine.fileio import get as get_
from mmdet.visualization import get_palette
from mmcv.image import imfrombytes, imwrite



def load_data_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


_VALID_PROMPT_MODES = ('text_only', 'text_visual', 'visual')
_MODE_FLAGS = frozenset({'present_only', 'solo'})

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def _parse_demo_mode(mode: str):
    """Split mode into core spec, flags, optional chunk size (demo-local parser).

    Suffixes: ``.present_only``, ``.solo`` (chunk=1), ``.chunk.N``.
    """
    if not mode:
        return '', (), None
    parts = mode.split('.')
    flags = []
    chunk_size = None
    while parts:
        if parts[-1] in _MODE_FLAGS:
            flags.insert(0, parts.pop())
            continue
        if len(parts) >= 2 and parts[-2] == 'chunk' and parts[-1].isdigit():
            chunk_size = int(parts.pop())
            parts.pop()
            continue
        break
    if 'solo' in flags:
        chunk_size = 1
    return '.'.join(parts), tuple(flags), chunk_size


def _mode_core(mode: str) -> str:
    return _parse_demo_mode(mode)[0]


def _mode_present_only(mode: str) -> bool:
    flags = _parse_demo_mode(mode)[1]
    if 'present_only' in flags:
        return True
    core = _mode_core(mode)
    base = core.split('.')[0] if core else ''
    return (
        base == 'visual' and 'I' in core.split('.')
        and os.getenv('NO_INTERACTIVE_CHUNK', '0') == '0')


def _mode_chunk_size(mode: str, default: int = -1) -> int:
    _, flags, chunk_size = _parse_demo_mode(mode)
    if 'solo' in flags:
        return 1
    if chunk_size is not None and chunk_size > 0:
        return chunk_size
    return default


def parse_text_prompt(texts: str) -> list:
    """Parse ``--texts``: ``person . car``, or ``$: lvis`` / ``$: coco``."""
    texts = texts.strip()
    if not texts:
        raise ValueError('--texts is empty.')
    if texts.startswith('$:'):
        return list(get_classes(texts[3:].strip()))
    parts = re.split(r'\s*\.\s*|\s*;\s*', texts)
    return [p.strip() for p in parts if p.strip()]


def _resolve_image_path(item: dict, data_prefix: str) -> str:
    file_name = item['file_name']
    if (os.path.isabs(file_name) or file_name.startswith(('http://', 'https://'))):
        return file_name
    if data_prefix:
        return os.path.join(data_prefix, file_name)
    return file_name


def parse_visual_prompts_spec(
        spec: str,
        entity_names: list,
) -> dict:
    """Parse visual.I prompts: JSON file, inline JSON, or per-image dict."""
    if spec is None or not str(spec).strip():
        return {}
    spec = spec.strip()
    if os.path.isfile(spec):
        with open(spec, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(spec)
        except json.JSONDecodeError:
            data = ast.literal_eval(spec)

    name_to_idx = {n: i for i, n in enumerate(entity_names)}

    def _norm_one(entry: dict) -> dict:
        cat = entry.get('category') or entry.get('label') or entry.get('name')
        if cat is None:
            raise ValueError(
                f'visual prompt entry missing category: {entry}')
        cat = str(cat)
        if cat not in name_to_idx:
            raise ValueError(
                f'category {cat!r} not in --texts entities {entity_names}')
        bbox = entry.get('bbox') or entry.get('box')
        if not bbox or len(bbox) != 4:
            raise ValueError(f'invalid bbox in visual prompt entry: {entry}')
        return {
            'bbox': [float(x) for x in bbox],
            'category': cat,
            'category_id': name_to_idx[cat],
            'ignore_flag': int(entry.get('ignore_flag', 0)),
        }

    if isinstance(data, list):
        return {'__all__': [_norm_one(e) for e in data]}
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if not isinstance(v, list):
                raise ValueError(f'visual prompts[{k!r}] must be a list.')
            out[k] = [_norm_one(e) for e in v]
        return out
    raise ValueError(
        '--visual-prompts must be a JSON list or dict (file path or inline).')


def visual_prompts_for_image(visual_map: dict, img_path: str) -> list:
    if not visual_map:
        return []
    if '__all__' in visual_map:
        return list(visual_map['__all__'])
    base = os.path.basename(img_path)
    if base in visual_map:
        return list(visual_map[base])
    if img_path in visual_map:
        return list(visual_map[img_path])
    return []


def list_image_files(paths) -> list:
    """Collect image paths from files and/or directories."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith(IMG_EXTENSIONS):
                        out.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            out.append(p)
        elif p.startswith(('s3://', 'nori://')):
            raise NotImplementedError(
                'Remote backends are not supported; use local paths.')
        elif p.startswith(('http://', 'https://')):
            out.append(p)
        else:
            print_log(f'[WARN] image path not found: {p}', logger='current')
    return out


def build_manual_items(
        image_paths: list,
        text_entities: list,
        visual_prompts_map: dict | None = None,
) -> tuple:
    """Build demo items from image(s) + manual text / visual.I prompts."""
    visual_prompts_map = visual_prompts_map or {}
    name_to_idx = {n: i for i, n in enumerate(text_entities)}
    items = []
    for idx, img_path in enumerate(image_paths):
        insts = visual_prompts_for_image(visual_prompts_map, img_path)
        for inst in insts:
            inst['bbox_label'] = name_to_idx[inst['category']]
        items.append({
            'img_id': idx,
            'file_name': img_path,
            'instances': insts,
            'text': list(text_entities),
            'manual_visual_prompts': bool(insts),
        })
    vocab = list(text_entities)
    cat_id_to_label = {i: i for i in range(len(vocab))}
    label_map = {n: i for i, n in enumerate(text_entities)}
    return items, label_map, vocab, cat_id_to_label


def collect_input_sources(args, modes: list) -> list:
    """Return inference sources for ``json`` or ``image`` subcommand."""
    sources = []
    cmd = getattr(args, 'command', None)

    if cmd == 'json':
        json_files = list_files(args.input, args.file_ext)
        json_files = [f for f in json_files if args.file_patten in f]
        if not json_files:
            raise ValueError(f'No JSON files matched: {args.input}')
        for json_file in json_files:
            items, _label_map, vocab, cat_id_to_label = load_items_auto(
                json_file)
            sources.append({
                'kind': 'json',
                'name': os.path.splitext(os.path.basename(json_file))[0],
                'items': items,
                'vocab': vocab,
                'cat_id_to_label': cat_id_to_label,
                'data_prefix': args.data_prefix,
            })

    elif cmd == 'image':
        image_paths = list(dict.fromkeys(list_image_files(args.input)))
        if not image_paths:
            raise ValueError(f'No images found under: {args.input}')
        text_entities = parse_text_prompt(args.texts)
        visual_map = {}
        if args.visual_prompts:
            visual_map = parse_visual_prompts_spec(
                args.visual_prompts, text_entities)
        for mode in modes:
            if 'I' in _mode_core(mode).split('.') and not visual_map:
                raise ValueError(
                    f'mode {mode!r} requires --visual-prompts for image input.')
        items, _label_map, vocab, cat_id_to_label = build_manual_items(
            image_paths, text_entities, visual_map)
        if len(image_paths) == 1:
            stem = os.path.splitext(os.path.basename(image_paths[0]))[0]
        elif len(args.input) == 1 and os.path.isdir(args.input[0]):
            stem = os.path.basename(os.path.normpath(args.input[0]))
        else:
            stem = 'manual_images'
        sources.append({
            'kind': 'image',
            'name': stem,
            'items': items,
            'vocab': vocab,
            'cat_id_to_label': cat_id_to_label,
            'data_prefix': '',
        })
    else:
        raise ValueError('Missing subcommand: use "json" or "image".')

    return sources


def add_common_args(parser: ArgumentParser) -> None:
    """Arguments shared by ``json`` and ``image`` subcommands."""
    parser.add_argument(
        '--mode',
        type=str,
        default=['text_only'],
        nargs='+',
        help='OPUS modes; suffix flags: .present_only .solo .chunk.N '
             '(e.g. text_visual.G.16.present_only.solo)')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument(
        '--chunked-size',
        type=int,
        default=-1,
        help='model.test_cfg.chunked_size; overridden by mode .solo / .chunk.N')
    parser.add_argument(
        '--randomness-seed',
        type=int,
        default=None,
        help='Seed for visual.I GT box pre-selection on JSON input.')
    parser.add_argument('--out-dir', type=str, default='./work_dirs/demo')
    parser.add_argument(
        '--gen-index',
        action='store_true',
        help=(
            'After inference, run tools/demo_vis_index.py to write '
            '{out-dir}/{experiment}.html for cross-mode vis compare.'
        ),
    )
    parser.add_argument(
        '--index-output-dir',
        type=str,
        default=None,
        help='HTML output directory for --gen-index (default: {out-dir}).',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Parallel workers (default: number of GPUs).')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.2,
        help='Score threshold for NMS post-process and visualization.')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for NMS (default class-agnostic).')
    parser.add_argument(
        '--class-aware-nms',
        action='store_true',
        help='Run per-class NMS instead of class-agnostic (default).')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument(
        '--vis-gt',
        action='store_true',
        help='Save side-by-side vis: left=GT, right=pred.')
    parser.add_argument(
        '--no-show-prompt',
        action='store_true',
        help='Do not draw text prompts under visualization images.')
    parser.add_argument(
        '--no-draw-label',
        action='store_true',
        help='Draw detection boxes only (no class/score text on boxes).')
    parser.add_argument('--print-result', action='store_true')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Vis palette (use random for LVIS / open-vocab).')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='PromptDetInferencer config .py')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Checkpoint .pth')


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='OPUS inference demo (visualization). JSON uses GT only to '
                    'pre-build prompts; image mode uses manual text/boxes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s json ann.json --data-prefix data/coco --mode text_only.present_only --save-vis
  %(prog)s json ann.json --mode text_visual.G.16.present_only.solo --save-vis
  %(prog)s image demo.jpg --texts "person . car" --mode text_only.solo
  %(prog)s image imgs/ --texts '$: lvis' --mode text_only.chunk.40
  %(prog)s image a.jpg --texts "person . cup" --mode visual.I.1 \\
      --visual-prompts '[{"category":"person","bbox":[10,20,100,200]}]'
""".strip())
    sub = parser.add_subparsers(dest='command', required=True)

    p_json = sub.add_parser(
        'json',
        help='Annotation JSON: pre-select present classes (present_only) and/or '
             'visual.I prompt boxes—not full-GT eval.')
    p_json.add_argument(
        'input',
        nargs='+',
        help='Annotation JSON file(s) or directory.')
    p_json.add_argument(
        '--data-prefix',
        type=str,
        default='',
        help='Image root prepended to COCO ``file_name`` (local path).')
    p_json.add_argument(
        '--file-ext',
        type=str,
        default='.json',
        help='Extension filter when traversing directories.')
    p_json.add_argument(
        '--file-patten',
        type=str,
        default='.json',
        help='Substring filter on discovered JSON paths.')
    p_json.add_argument(
        '--sample-num-limit',
        type=int,
        default=-1,
        help='Max samples per JSON source (-1 = all).')
    p_json.add_argument(
        '--save-pred',
        action='store_true',
        help='Merge batch NMS outputs to preds/*.pkl (debug dump, not AP eval).')
    add_common_args(p_json)

    p_image = sub.add_parser(
        'image',
        help='Visualize detections on image(s) with manual text & visual.I prompts.')
    p_image.add_argument(
        'input',
        nargs='+',
        help='Image path(s), directory, or local directory.')
    p_image.add_argument(
        '--texts',
        type=str,
        required=True,
        help='Class names: "person . car" or "$: lvis" / "$: coco".')
    p_image.add_argument(
        '--visual-prompts',
        type=str,
        default=None,
        help='visual.I boxes: JSON file or inline list/dict.')
    p_image.add_argument(
        '--no-save-vis',
        dest='save_vis',
        action='store_false',
        help='Disable saving visualization images (on by default).')
    add_common_args(p_image)
    p_image.set_defaults(save_vis=True)

    return parser


def normalize_demo_modes(modes):
    """Validate OPUS mode strings (optional ``.present_only`` / ``.solo`` / ``.chunk.N``)."""
    normalized = []
    for mode in modes:
        mode = mode.strip()
        if not mode:
            continue
        base = _mode_core(mode).split('.')[0] if mode else ''
        if base not in _VALID_PROMPT_MODES:
            raise ValueError(
                f'Unsupported mode {mode!r} (base {base!r}); '
                f'expected one of {_VALID_PROMPT_MODES}')
        chunk_size = _mode_chunk_size(mode, default=-1)
        if chunk_size == 0:
            raise ValueError(
                f'Invalid chunk size in mode {mode!r}; use .solo or .chunk.N with N>=1')
        normalized.append(mode)
    if not normalized:
        raise ValueError('At least one --mode is required.')
    return normalized


def validate_demo_modes(modes, prompt_path: str) -> None:
    for mode in modes:
        if 'G' in _mode_core(mode).split('.') and not prompt_path:
            raise ValueError(
                f'mode {mode!r} requires --prompt-path (generic visual embed pkl).')

def list_files(path, file_ext=None):
    res = []
    if isinstance(path, str):
        path = [path]
    for p in path:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for file in files:
                    if file_ext is None or file.endswith(file_ext):
                        res.append(os.path.join(root, file))
        elif os.path.isfile(p) and (file_ext is None or p.endswith(file_ext)):
            res.append(p)
    return res

def load_items_auto(json_file: str, min_size: int = 1):
    """
    自动加载 COCO / OD / VG 格式的数据，并转换为每张图片一条 item
    每条 item 结构：
        {
            'img_id': ...,
            'file_name': ...,
            'instances': [ {'bbox': [x1,y1,x2,y2], 'category_id': int, 'category': str, 'ignore_flag': 0}, ...],
            'text': [该图 GT 出现过的类名, ...],  # present_only
            # LVIS 联邦（有 neg_category_ids 时）：
            'federated_text': [pos ∪ neg 类名, ...],  # 与 LVISEval 评测范围一致
        }
    返回：
        items: List[item]
        label_map: dict, 类名 -> category_id
        vocab: List[str], categories 全表（按 category id 排序；COCO 全词表 fallback）
        cat_id_to_label: dict, category_id -> vocab 下标
    """
    try:
        data = load_data_from_json(json_file)
    except Exception:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = [json.load(f)]

    items = []
    label_set = set()
    vocab: list = []
    cat_id_to_label: dict = {}
    item = data[0]
    # ---------- 1) COCO / OD ----------
    if isinstance(item, dict) and "images" in item and "annotations" in item:
        print("[INFO] Detected COCO/OD format")
        imgs = item.get("images", [])
        anns = item.get("annotations", [])
        categories = {i["id"]: i["name"] for i in item.get("categories", [])}
        cat_rows = sorted(item.get("categories", []), key=lambda c: c["id"])
        vocab = [c["name"] for c in cat_rows]
        cat_id_to_label = {c["id"]: idx for idx, c in enumerate(cat_rows)}
        # LVIS federated: images carry explicit neg_category_ids (P∪N eval scope).
        is_lvis_federated = any(
            isinstance(im.get('neg_category_ids'), list)
            for im in imgs[: min(32, len(imgs))])
        if is_lvis_federated:
            print(
                "[INFO] LVIS federated ann detected: non-present_only prompts "
                "use per-image pos∪neg_category_ids (not full vocab)")
        # 建立 image_id -> annotations 映射
        img_id2anns = defaultdict(list)
        for ann in anns:
            if ann.get('ignore', False) or ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if ann['area'] <= 0 or w < min_size or h < min_size:
                continue
            ann['bbox'] = [x, y, x + w, y + h]
            img_id2anns[ann['image_id']].append(ann)
        for img_info in imgs:
            img_id = img_info.get("id")
            img_anns = img_id2anns.get(img_id, [])
            insts = []
            cat_names = []
            pos_cat_ids = set()
            for ann in img_anns:
                cid = ann.get("category_id")
                cat_name = categories[cid]
                insts.append({
                    'bbox': ann['bbox'],
                    'category': cat_name,
                    'category_id': cid,
                    'ignore_flag': 0,
                })
                cat_names.append(cat_name)
                pos_cat_ids.add(cid)
                label_set.add(cat_name)
            if not insts:
                continue
            row = {
                'img_id': img_id,
                'file_name': img_info.get("file_name", img_info.get("filename")),
                'instances': insts,
                'text': list(set(cat_names)),
            }
            if is_lvis_federated:
                neg_ids = {
                    int(x) for x in (img_info.get('neg_category_ids') or [])
                    if int(x) in categories
                }
                # Match LVISEval scope for this image: annotated pos ∪ explicit neg.
                fed_ids = sorted(pos_cat_ids | neg_ids)
                row['federated_text'] = [categories[cid] for cid in fed_ids]
            items.append(row)
    # ---------- 2) OD 格式 ----------
    elif isinstance(item, dict) and "detection" in item:
        print("[INFO] Detected OD format")
        for idx, item in enumerate(data):
            img_id = item.get("image_id", idx)
            file_name = item.get("filename", item.get("file_name", f"{img_id}.jpg"))

            # 支持 detection.instances
            det = item.get("detection", {})
            instances_raw = det.get("instances", [])
            insts = []
            cat_names = []

            for ann in instances_raw:
                cat_name = ann.get("category") or ann.get("label") or "__unknown__"
                cat_names.append(str(cat_name))
                label_set.add(str(cat_name))
                if not isinstance(ann, dict):
                    continue
                bbox = ann.get('bbox', [0,0,0,0])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue
                insts.append({
                    'bbox': [x1, y1, x2, y2],
                    'category': str(cat_name),
                    'ignore_flag':0
                })

            if not insts:
                continue

            items.append({
                'img_id': img_id,
                'file_name': file_name,
                'instances': insts,
                'text': list(set(cat_names))
            })
    # ---------- 3) VG 格式 ----------
    elif isinstance(item, dict) and "grounding" in item:  # VG
        print("[INFO] Detected VG format")
        for idx, item in enumerate(data):
            grounding = item.get("grounding", {})
            caption = grounding.get("caption", "")
            img_id = item.get("image_id", idx)
            file_name = item.get("filename", item.get("file_name", f"{img_id}.jpg"))

            insts = []
            entity_set = set()

            for region in grounding.get("regions", []):
                tokens = region.get("tokens_positive", [])
                sorted_tokens = sorted(tokens, key=lambda x: x[0])
                prev = -10
                entities = []
                for (s, e) in sorted_tokens:
                    if prev > 0 and prev < s and not caption[prev:s].replace(" ", ""):
                        entities[-1] += f" {caption[s:e]}"
                    elif prev == s:
                        entities[-1] += f"{caption[s:e]}"
                    else:
                        entities.append(caption[s:e])
                    prev = e

                label = ", ".join(entities)
                if not label:
                    label = "__unknown__"
                entity_set.add(label)
                label_set.add(label)
                
                bbox = region.get("bbox", [0,0,0,0])
                if not isinstance(bbox[0], list):
                    bbox = [bbox]
                for box in bbox:
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    # 每个 region 作为一个 instance
                    insts.append({
                        "bbox": box, 
                        "category": label,
                        'ignore_flag':0
                    })

            if insts:
                items.append({
                    "img_id": img_id,
                    "file_name": file_name,
                    "instances": insts,
                    "text": list(entity_set)
                })
    else:
        raise ValueError(f"Unknown dataset format: {json_file}")

    # 构建 label_map（OD/VG 等无 categories 表时回退为 sorted names）
    label_list = sorted(label_set)
    label_map = {name: idx for idx, name in enumerate(label_list)}
    if not vocab:
        vocab = label_list
        name_to_label = label_map
        for item_row in items:
            for inst in item_row.get('instances', []):
                if 'category_id' not in inst:
                    inst['category_id'] = name_to_label.get(inst['category'])

    n_fed = sum(1 for it in items if it.get('federated_text'))
    print(
        f"[INFO] Loaded {len(items)} items, vocab={len(vocab)}, "
        f"unique_names={len(label_map)}, federated_prompt_imgs={n_fed} "
        f"from {json_file}")
    return items, label_map, vocab, cat_id_to_label


def _instance_bbox_label(
        inst: dict,
        *,
        prompt_names: list,
        use_global_vocab: bool,
        cat_id_to_label: dict,
) -> int:
    """Class index into ``prompt_names`` (or global vocab when ``use_global_vocab``)."""
    if use_global_vocab:
        cat_id = inst.get('category_id')
        if cat_id is not None and cat_id in cat_id_to_label:
            return cat_id_to_label[cat_id]
        return prompt_names.index(inst['category'])
    return prompt_names.index(inst['category'])


def _prompt_names_for_item(
        item: dict,
        *,
        present_only: bool,
        vocab: list,
) -> tuple:
    """Return ``(prompt_names, use_global_vocab)`` for one image.

    Priority:
      1. ``present_only`` → GT-present classes (``item['text']``)
      2. LVIS federated → ``pos ∪ neg_category_ids`` (``item['federated_text']``)
      3. else global ``vocab`` (e.g. COCO 80) when available
    """
    if present_only:
        return list(item.get('text') or []), False
    fed = item.get('federated_text')
    if fed:
        return list(fed), False
    if vocab:
        return list(vocab), True
    return list(item.get('text') or []), False


def _interactive_instances_preselected(
        item: dict,
        *,
        prompt_names: list,
        use_global_vocab: bool,
        cat_id_to_label: dict,
        visual_prompts_num: int | None,
        seed: int | None = None,
) -> list:
    """Pre-select visual prompt boxes for ``visual.I*`` (``N`` per class).

    Demo intentionally samples from JSON GT *before* inference so the model
    only sees the chosen prompt boxes—not the full annotation set.  This
    matches an interactive inference UX (user picks exemplars) rather than
    the eval path that passes all ``gt_instances`` into ``predict``.
    """
    by_label: dict = defaultdict(list)
    for inst in item.get('instances', []):
        cat = inst.get('category')
        if cat is None:
            continue
        if not use_global_vocab and cat not in prompt_names:
            continue
        inst_copy = inst.copy()
        inst_copy['bbox_label'] = _instance_bbox_label(
            inst_copy,
            prompt_names=prompt_names,
            use_global_vocab=use_global_vocab,
            cat_id_to_label=cat_id_to_label,
        )
        by_label[inst_copy['bbox_label']].append(inst_copy)

    rng = np.random.default_rng(seed)
    selected = []
    for group in by_label.values():
        n = len(group)
        k = visual_prompts_num if visual_prompts_num else n
        k = min(int(k), n)
        if k <= 0:
            continue
        pick = rng.choice(n, size=k, replace=False)
        selected.extend(group[i] for i in pick)
    return selected


def _append_prompt_banner(
        img_bgr: np.ndarray,
        prompt_names: list,
        *,
        mode: str = '',
        palette: str = 'random',
        detected_labels=None,
        max_lines: int = 8,
) -> np.ndarray:
    """Pad a white banner under ``img_bgr``.

    Detected classes use the same palette colors as boxes; others are black.
    """
    names = [str(n).strip() for n in (prompt_names or []) if str(n).strip()]
    if (not names and not mode) or img_bgr is None or img_bgr.size == 0:
        return img_bgr
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img_bgr

    h, w = img_bgr.shape[:2]
    font_size = max(14, min(28, w // 40))
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except OSError:
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                font_size)
        except OSError:
            font = ImageFont.load_default()

    n_cls = max(len(names), 1)
    palette_rgb = get_palette(
        palette if palette != 'none' else 'random', n_cls)
    if detected_labels is None:
        detected = set(range(len(names)))
    else:
        detected = {int(x) for x in detected_labels}
    # Tokens: (text, RGB fill). Same index→color as detection boxes when hit.
    tokens = []
    if mode:
        tokens.append((f'[{mode}] ', (20, 20, 20)))
    for i, name in enumerate(names):
        if i:
            tokens.append((' . ', (120, 120, 120)))
        if i in detected:
            rgb = palette_rgb[i % len(palette_rgb)][:3]
            color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        else:
            color = (0, 0, 0)
        tokens.append((name, color))

    pad_x = max(8, w // 80)
    line_gap = max(4, font_size // 4)
    draw_probe = ImageDraw.Draw(Image.new('RGB', (w, 8)))
    max_w = w - 2 * pad_x

    def _text_w(s: str) -> int:
        bbox = draw_probe.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    def _text_h(s: str) -> int:
        bbox = draw_probe.textbbox((0, 0), s, font=font)
        return bbox[3] - bbox[1]

    # Greedy wrap keeping tokens intact when possible.
    lines = []  # list[list[(text, color)]]
    cur, cur_w = [], 0
    for text, color in tokens:
        tw = _text_w(text)
        if cur and cur_w + tw > max_w:
            lines.append(cur)
            cur, cur_w = [], 0
            if len(lines) >= max_lines:
                break
        # Truncate oversized single token.
        if tw > max_w and not cur:
            s = text
            while len(s) > 1 and _text_w(s + '…') > max_w:
                s = s[:-1]
            cur.append((s + '…', color))
            lines.append(cur)
            cur, cur_w = [], 0
            if len(lines) >= max_lines:
                break
            continue
        cur.append((text, color))
        cur_w += tw
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if not lines:
        return img_bgr

    line_heights = [
        max((_text_h(t) for t, _ in line), default=font_size) for line in lines
    ]
    banner_h = (
        pad_x + sum(line_heights) + line_gap * max(0, len(lines) - 1) + pad_x)
    banner_h = max(banner_h, font_size + 2 * pad_x)

    # PIL draws in RGB; convert BGR↔RGB around the banner.
    img_rgb = img_bgr[:, :, ::-1]
    canvas = np.full((h + banner_h, w, 3), 255, dtype=np.uint8)
    canvas[:h] = img_rgb
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    y = h + pad_x
    for line, lh in zip(lines, line_heights):
        x = pad_x
        for text, color in line:
            draw.text((x, y), text, fill=color, font=font)
            x += _text_w(text)
        y += lh + line_gap
    return np.asarray(pil)[:, :, ::-1]


def _as_numpy_inds(inds):
    if hasattr(inds, 'detach'):
        inds = inds.detach().cpu().numpy()
    return np.asarray(inds)


def _nms_filter(
        bboxes,
        scores,
        labels,
        score_thr: float,
        iou_thr: float,
        class_agnostic: bool = True,
):
    """Score filter + NMS. Default ``class_agnostic=True`` (mmcv ``nms``)."""
    if bboxes is None or len(bboxes) == 0:
        return [], [], []
    boxes = np.asarray(bboxes, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    lab = np.asarray(labels)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return [], [], []
    keep_score = sc >= float(score_thr)
    boxes, sc, lab = boxes[keep_score], sc[keep_score], lab[keep_score]
    if boxes.shape[0] == 0:
        return [], [], []

    def _run_nms(b, s):
        try:
            out = nms(b, s, iou_threshold=float(iou_thr), score_threshold=0.0)
        except TypeError:
            out = nms(b, s, iou_threshold=float(iou_thr))
        if isinstance(out, (tuple, list)):
            return _as_numpy_inds(out[1] if len(out) > 1 else out[0])
        return _as_numpy_inds(out)

    if class_agnostic:
        inds = _run_nms(boxes, sc)
    else:
        keep = []
        for c in np.unique(lab):
            mask = lab == c
            if not np.any(mask):
                continue
            local = _run_nms(boxes[mask], sc[mask])
            if local.size == 0:
                continue
            global_idx = np.flatnonzero(mask)[local]
            keep.extend(global_idx.tolist())
        if not keep:
            return [], [], []
        # Re-sort kept by score descending for stable vis.
        inds = np.asarray(keep)[np.argsort(-sc[np.asarray(keep)])]
    if inds.size == 0:
        return [], [], []
    return boxes[inds].tolist(), sc[inds].tolist(), lab[inds].tolist()


def _draw_det_panel(
        img_bgr: np.ndarray,
        bboxes,
        labels,
        class_names: list,
        scores=None,
        *,
        title: str = '',
        palette: str = 'random',
        line_width: int = 2,
        draw_label: bool = True,
) -> np.ndarray:
    """Draw boxes/labels on a BGR copy of ``img_bgr``."""
    out = img_bgr.copy()
    if cv2 is None:
        return out
    n_cls = max(len(class_names), 1)
    max_lab = 0
    if labels is not None and len(labels):
        max_lab = int(max(int(x) for x in labels))
    # get_palette returns RGB tuples; convert to BGR for cv2 drawing.
    colors = get_palette(palette if palette != 'none' else 'random',
                         max(n_cls, max_lab + 1))

    if title:
        cv2.putText(
            out, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            out, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 0), 2, cv2.LINE_AA)

    if not bboxes:
        return out
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        lab = int(labels[i]) if labels is not None else 0
        rgb = colors[lab % len(colors)][:3]
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # RGB -> BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)
        if not draw_label:
            continue
        if class_names and 0 <= lab < len(class_names):
            name = str(class_names[lab])
        else:
            name = f'class {lab}'
        if scores is not None and i < len(scores):
            name = f'{name}: {float(scores[i]) * 100:.1f}'
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 4)
        x0, y0 = x1, max(0, ty - th - 4)
        x1b, y1b = min(out.shape[1], x1 + tw + 4), min(out.shape[0], ty + 2)
        if x1b > x0 and y1b > y0:
            # Lower transparency vs previous 0.35 (≈ half as see-through).
            label_bg_alpha = 0.7
            roi = out[y0:y1b, x0:x1b]
            overlay = np.zeros_like(roi)
            out[y0:y1b, x0:x1b] = cv2.addWeighted(
                overlay, label_bg_alpha, roi, 1.0 - label_bg_alpha, 0)
        cv2.putText(
            out, name, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA)
    return out


def _gt_boxes_labels(item: dict, prompt_names: list):
    """Collect GT boxes; labels index into ``prompt_names`` when possible."""
    bboxes, labels, extra_names = [], [], list(prompt_names)
    name_to_id = {n: i for i, n in enumerate(extra_names)}
    for inst in item.get('instances', []) or []:
        cat = inst.get('category')
        box = inst.get('bbox')
        if cat is None or box is None or len(box) != 4:
            continue
        if cat not in name_to_id:
            name_to_id[cat] = len(extra_names)
            extra_names.append(cat)
        bboxes.append([float(x) for x in box])
        labels.append(name_to_id[cat])
    return bboxes, labels, extra_names


def _save_demo_visualization(
        img_bgr: np.ndarray,
        *,
        pred_bboxes,
        pred_labels,
        pred_scores,
        prompt_names: list,
        item: dict,
        out_path: str,
        mode: str,
        vis_gt: bool,
        show_prompt: bool,
        draw_label: bool,
        palette: str,
):
    pred_panel = _draw_det_panel(
        img_bgr,
        pred_bboxes,
        pred_labels,
        prompt_names,
        pred_scores,
        title='',
        palette=palette,
        draw_label=draw_label,
    )
    if vis_gt:
        gt_bboxes, gt_labels, gt_names = _gt_boxes_labels(item, prompt_names)
        gt_panel = _draw_det_panel(
            img_bgr,
            gt_bboxes,
            gt_labels,
            gt_names,
            None,
            title='GT',
            palette=palette,
            draw_label=draw_label,
        )
        h = max(gt_panel.shape[0], pred_panel.shape[0])

        def _pad(im):
            if im.shape[0] == h:
                return im
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
            return np.concatenate([im, pad], axis=0)

        canvas = np.concatenate([_pad(gt_panel), _pad(pred_panel)], axis=1)
    else:
        canvas = pred_panel

    if show_prompt:
        canvas = _append_prompt_banner(
            canvas,
            prompt_names,
            mode=mode or '',
            palette=palette,
            detected_labels=pred_labels,
        )

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # imfrombytes(pillow) already returns BGR; mmcv.imwrite expects BGR.
    imwrite(canvas, out_path)


# -----------------------------
# run_batch_infer: 执行单个 batch 的推理并返回 embeddings dict
# -----------------------------
def run_batch_infer(task_param, inferencer, data_prefix=''):
    """
    task_param:
      {
        'json_file', 'mode', 'cat_name',
        'items' (list of item dicts for this batch),
        'batch_id'
      }
    Returns:
      dict: {predictions: [list of bboxes],...}
    """
    mode = task_param['mode']
    items = task_param['items']
    data_prefix = task_param.get('data_prefix', data_prefix)
    vocab = task_param.get('vocab') or []
    cat_id_to_label = task_param.get('cat_id_to_label') or {}
    prompt_path = task_param.get('prompt_path') or None
    core_mode = _mode_core(mode)
    present_only = _mode_present_only(mode)
    chunk_size = _mode_chunk_size(
        mode, int(task_param.get('chunked_size', -1)))
    use_interactive = 'I' in core_mode.split('.')
    visual_match = re.search(r'\.(\d+)$', core_mode)
    visual_prompts_num = int(visual_match.group(1)) if visual_match else None
    # Demo pre-filters prompts; inferencer gets core spec only (no suffix flags).
    infer_mode = core_mode
    if chunk_size > 0:
        inferencer.model.test_cfg.chunked_size = chunk_size
    else:
        inferencer.model.test_cfg.chunked_size = int(
            task_param.get('chunked_size', -1))
    if present_only and chunk_size > 0 and len(items) > 1:
        print_log(
            '[WARN] present_only + chunked mode with batch_size>1 may disable '
            'chunking inside the model; use --batch-size 1 for solo compare.',
            logger='current',
            level='WARNING')
    pass_instances = use_interactive
    score_thr = float(task_param.get('score_thr', 0.2))
    iou_thr = float(task_param.get('iou_thr', 0.5))
    class_agnostic = bool(task_param.get('class_agnostic_nms', True))
    save_merged_pred = task_param.get('save_merged_pred', False)
    save_vis = task_param.get('save_vis', False)
    vis_gt = task_param.get('vis_gt', False)
    show_prompt = task_param.get('show_prompt', True)
    draw_label = task_param.get('draw_label', True)
    print_result = task_param.get('print_result', False)
    out_dir = task_param.get('out_dir', None)
    palette = task_param.get('palette', 'random')
    batch_id = task_param['batch_id']
    batch_size = task_param['batch_size']

    sample_inputs, sample_texts, sample_instances, sample_meta = [], [], [], []

    for item in items:
        img_path = _resolve_image_path(item, data_prefix)
        try:
            img_bytes = get_(img_path)
            img = imfrombytes(img_bytes, flag='color', backend='pillow')
        except Exception as e:
            print(f"[WARN] Failed to read {img_path}: {e}")
            continue

        prompt_names, use_global_vocab = _prompt_names_for_item(
            item, present_only=present_only, vocab=vocab)
        if not prompt_names:
            print(f"[WARN] Empty prompt set for {img_path}, skip")
            continue

        instances = []
        if use_interactive:
            if item.get('manual_visual_prompts'):
                for inst in item.get('instances', []):
                    inst_copy = inst.copy()
                    try:
                        inst_copy['bbox_label'] = _instance_bbox_label(
                            inst_copy,
                            prompt_names=prompt_names,
                            use_global_vocab=use_global_vocab,
                            cat_id_to_label=cat_id_to_label,
                        )
                    except ValueError:
                        continue
                    instances.append(inst_copy)
            else:
                item_seed = None
                base_seed = task_param.get('randomness_seed')
                if base_seed is not None:
                    item_seed = int(base_seed) + int(item.get('img_id', 0))
                instances = _interactive_instances_preselected(
                    item,
                    prompt_names=prompt_names,
                    use_global_vocab=use_global_vocab,
                    cat_id_to_label=cat_id_to_label,
                    visual_prompts_num=visual_prompts_num,
                    seed=item_seed,
                )
            if not instances:
                continue

        sample_inputs.append(img)
        sample_texts.append(prompt_names)
        sample_instances.append(instances)
        sample_meta.append({'item': item, 'img_path': img_path})

    if not sample_inputs:
        return {}

    # Inference only — visualization is done below, not via Inferencer.
    call_args = {
        'inputs': sample_inputs,
        'texts': sample_texts,
        'mode': infer_mode,
        'prompt_path': prompt_path,
        'instances': sample_instances if pass_instances else None,
        'out_dir': '',
        'batch_size': len(sample_inputs),
        'no_save_vis': True,
        'no_save_pred': True,
        'print_result': print_result,
        'custom_entities': True,
        'tokens_positive': None,
        'show': False,
    }

    results = inferencer(**call_args)
    predictions = results.get('predictions', [])
    if len(predictions) != len(sample_inputs):
        print(
            f"[WARN] pred count {len(predictions)} != inputs "
            f"{len(sample_inputs)}")

    batch_results = defaultdict(list)
    vis_dir = os.path.join(out_dir, 'vis') if (save_vis and out_dir) else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    for i, pred in enumerate(predictions):
        if i >= len(sample_meta):
            break
        meta = sample_meta[i]
        prompt_names = sample_texts[i]
        bboxes, scores, labels = _nms_filter(
            pred.get('bboxes', []),
            pred.get('scores', []),
            pred.get('labels', []),
            score_thr=score_thr,
            iou_thr=iou_thr,
            class_agnostic=class_agnostic,
        )
        if save_vis and vis_dir is not None:
            img_name = f'{str(batch_id * batch_size + i).zfill(8)}.jpg'
            _save_demo_visualization(
                sample_inputs[i],
                pred_bboxes=bboxes,
                pred_labels=labels,
                pred_scores=scores,
                prompt_names=prompt_names,
                item=meta['item'],
                out_path=os.path.join(vis_dir, img_name),
                mode=mode,
                vis_gt=vis_gt,
                show_prompt=show_prompt,
                draw_label=draw_label,
                palette=palette,
            )
        if save_merged_pred:
            batch_results['bboxes'].append(bboxes)
            batch_results['scores'].append(scores)
            batch_results['labels'].append(labels)
            if 'embeddings' in pred:
                for k, v in pred['embeddings'].items():
                    batch_results[k].extend(v)

    return dict(batch_results) if save_merged_pred else {}

# -----------------------------
# gpu_worker: 每个 GPU 上跑一个 worker，从队列拉任务 -> 推理 -> 写回 result_queue
# -----------------------------
def gpu_worker(task_queue: Queue,
               result_queue: Queue,
               progress_counter: Value,
               progress_lock: Lock,
               worker_id: int,
               model_cfg: str,
               weights_path: str,
               data_prefix: str = '',
               chunked_size: int = -1,
               palette: str = 'random'):
    """
    每个进程启动一个 PromptDetInferencer 并循环拉任务。
    result_queue: 用于将 (task_id, json_file, mode, batch_results) 发送回主进程
    progress_counter: multiprocessing.Value (int) - 已完成 batch 计数
    progress_lock: Lock protecting increment of progress_counter
    """
    try:
        device = f'cuda:{worker_id}' if torch.cuda.is_available() else None
        inferencer = PromptDetInferencer(
            model=model_cfg,
            weights=weights_path,
            device=device,
            palette=palette,
            show_progress=False)
        inferencer.model.test_cfg.chunked_size = chunked_size
    except Exception as e:
        print(f"[ERROR] worker {worker_id} failed to init inferencer: {e}")
        traceback.print_exc()
        inferencer = None

    while True:
        try:
            task_param = task_queue.get(timeout=5)  # 超时退出
        except Exception:
            break

        try:
            if inferencer is None:
                batch_results = {}
            else:
                batch_results = run_batch_infer(task_param, inferencer, data_prefix=data_prefix)
        except Exception as e:
            print(f"[ERROR][worker {worker_id}] Exception while inferencing: {e}")
            traceback.print_exc()
            batch_results = {}

        # 发送到结果队列：包含 task_id, json_file, mode, batch_results
        try:
            res_obj = task_param
            res_obj['batch_results'] = batch_results
            result_queue.put(res_obj)
            # debug print
            if batch_results:
                total = len(list(batch_results.values())[0])
            else:
                total = 0
            print(f"[worker {worker_id}] finished task {res_obj['task_id']} count={total}")
        except Exception as e:
            print(f"[ERROR][worker {worker_id}] Failed to put result into queue: {e}")
            traceback.print_exc()

        # 更新进度计数（在 progress_lock 下）
        if progress_counter is not None:
            with progress_lock:
                progress_counter.value += 1

    # 结束
    print(f"[worker {worker_id}] exiting.")

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model = args.model
    weights = args.weights
    data_prefix = getattr(args, 'data_prefix', '')
    try:
        modes = normalize_demo_modes(args.mode)
    except ValueError as exc:
        print_log(str(exc), logger='current', level='ERROR')
        return
    try:
        validate_demo_modes(modes, args.prompt_path)
    except ValueError as exc:
        print_log(str(exc), logger='current', level='ERROR')
        return
    try:
        sources = collect_input_sources(args, modes)
    except ValueError as exc:
        print_log(str(exc), logger='current', level='ERROR')
        return

    # workers 数量（每个 GPU 一个 worker）
    if torch.cuda.is_available():
        args.num_workers = torch.cuda.device_count()
        print(f"Start processing with {args.num_workers} workers across {args.num_workers} GPUs.")
    else:
        print(f"Start processing with {args.num_workers} workers, CPU only.")

    # 任务队列与结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 生成任务：source -> mode -> batch
    total_batches = 0
    task_id = 0
    save_merged_pred = (
        args.command == 'json' and getattr(args, 'save_pred', False))
    for source in sources:
        items = source['items']
        vocab = source['vocab']
        cat_id_to_label = source['cat_id_to_label']
        data_prefix = source['data_prefix']
        filename = source['name']
        if source['kind'] == 'json':
            sample_num_limit = (
                args.sample_num_limit if args.sample_num_limit > 0
                else len(items))
        else:
            sample_num_limit = len(items)
        for mode in modes:
            subdir_1 = mode
            if weights is not None:
                subdir_2 = (os.path.normpath(weights).split(os.sep)[-2]
                            if '/' in weights or '\\' in weights
                            else os.path.splitext(
                                os.path.basename(weights))[0])
            else:
                subdir_2 = 'clip_raw'
            out_dir = os.path.join(args.out_dir, subdir_1, subdir_2)
            out_sub_dir = os.path.join(out_dir, 'samples', filename)
            if args.save_vis:
                os.makedirs(out_sub_dir, exist_ok=True)
            out_pickle_path = None
            if save_merged_pred:
                preds_dir = os.path.join(out_dir, 'preds')
                os.makedirs(preds_dir, exist_ok=True)
                out_pickle_path = os.path.join(preds_dir, f'{filename}.pkl')

            batch_size = args.batch_size
            for i in range(0, sample_num_limit, batch_size):
                batch_items = [items[item_id] for item_id in range(i, i+batch_size)]
                task = {
                    'task_id': task_id,
                    'batch_id': i // batch_size,
                    'batch_size': batch_size,
                    'items': batch_items,
                    'out_pickle_path': out_pickle_path,
                    'mode': mode,
                    'vocab': vocab,
                    'cat_id_to_label': cat_id_to_label,
                    'data_prefix': data_prefix,
                    'prompt_path': args.prompt_path or None,
                    'save_vis': args.save_vis,
                    'vis_gt': args.vis_gt,
                    'show_prompt': not args.no_show_prompt,
                    'draw_label': not args.no_draw_label,
                    'save_merged_pred': save_merged_pred,
                    'print_result': args.print_result,
                    'out_dir': out_sub_dir,
                    'score_thr': args.score_thr,
                    'iou_thr': args.iou_thr,
                    'class_agnostic_nms': not args.class_aware_nms,
                    'palette': args.palette,
                    'randomness_seed': args.randomness_seed,
                    'chunked_size': args.chunked_size,
                }
                task_queue.put(task)
                total_batches += 1
                task_id += 1

    print(f"[INFO] Total batches queued: {total_batches}")

    # 多进程构件：不使用 Manager.dict，使用 result_queue 汇总
    progress_counter = Value('i', 0)  # 已完成 batch 计数
    progress_lock = Lock()

    # 启动 GPU workers（每个 GPU 启动一个 process）
    workers = []
    for wid in range(args.num_workers):
        p = Process(target=gpu_worker, args=(
            task_queue, result_queue, progress_counter, progress_lock,
            wid, model, weights, data_prefix, args.chunked_size,
            args.palette))
        p.start()
        workers.append(p)

    merged_results = {} 
    processed = 0

    with tqdm(total=total_batches, desc='Batches processed') as pbar:
        while processed < total_batches:
            try:
                res = result_queue.get(timeout=1.0)
            except Exception:
                continue

            if save_merged_pred:
                key = res.get('out_pickle_path')
                batch_results = res.get('batch_results', {})
                if key not in merged_results:
                    merged_results[key] = {}
                for k, v in batch_results.items():
                    merged_results[key].setdefault(k, [])
                    merged_results[key][k].extend(v)

            processed += 1
            pbar.update(1)
            with progress_lock:
                progress_counter.value = processed

    for p in workers:
        p.join()

    if save_merged_pred:
        for out_pickle_path, results in merged_results.items():
            mmengine.dump(dict(results), out_pickle_path)
            total_counts = [(k, len(v)) for k, v in results.items()]
            print(f'[INFO] Saved merged debug preds to {out_pickle_path}')
            print(f'[INFO] Total: {total_counts}')

    if args.gen_index:
        index_script = os.path.join(os.path.dirname(__file__), 'demo_vis_index.py')
        cmd = [sys.executable, index_script, '--demo-dir', args.out_dir]
        if args.index_output_dir:
            cmd.extend(['--output-dir', args.index_output_dir])
        print(f'[INFO] Generating demo index: {" ".join(cmd)}')
        subprocess.run(cmd, check=False)

    print("[INFO] All done.")
    # clean exit
    try:
        # close queues
        task_queue.close()
        result_queue.close()
    except Exception:
        pass
    os._exit(0)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

