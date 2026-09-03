#!/usr/bin/env python3
"""Merge same-image entries in Flickr30k / GQA ODVG VG jsonlines.

Flickr30k and GQA often have multiple captions (lines) per image. This script
mirrors ``--merge-same-image`` in odvg-dataflow's ``odvg_2_raw_data.py``:

* group by ``filename``
* join captions with ``" .\\n "``
* shift each region's ``tokens_positive`` by the caption offset
* merge regions that share the same phrase (append bboxes)
* drop invalid / empty-box regions and images with no valid region

Input / output are ODVG VG jsonlines (one dict per line), as used by
``ODVGDataset`` and ``tools/dataset_converters/goldg2odvg.py``.

Examples::

  # Flickr30k → configs use flickr_30k_vg.json
  python tools/dataset_converters/merge_odvg_same_image.py \\
    --dataset flickr30k \\
    --input data/flickr30k_entities/final_flickr_separateGT_train_vg.json \\
    --output data/flickr30k_entities/flickr_30k_vg.json

  # GQA → configs use gqa_46k_vg.json
  python tools/dataset_converters/merge_odvg_same_image.py \\
    --dataset gqa \\
    --input data/gqa/final_mixed_train_no_coco_vg.json \\
    --output data/gqa/gqa_46k_vg.json

  # Generic path (no preset)
  python tools/dataset_converters/merge_odvg_same_image.py \\
    --input path/to/xxx_vg.json --output path/to/xxx_merged_vg.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CAPTION_SEP = ' .\n '

# Presets aligned with configs/datasets/640_640.py
DATASET_PRESETS = {
    'flickr30k': {
        'default_input': 'data/flickr30k_entities/final_flickr_separateGT_train_vg.json',
        'default_output': 'data/flickr30k_entities/flickr_30k_vg.json',
    },
    'gqa': {
        'default_input': 'data/gqa/final_mixed_train_no_coco_vg.json',
        'default_output': 'data/gqa/gqa_46k_vg.json',
    },
}


def _is_xyxy(box: Sequence[Any]) -> bool:
    return (
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(isinstance(x, (int, float)) for x in box))


def _normalize_bboxes(raw: Any,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> List[List[float]]:
    """Return valid xyxy boxes; drop empty / out-of-image boxes."""
    if raw is None:
        return []
    if _is_xyxy(raw):
        boxes = [list(map(float, raw))]
    elif isinstance(raw, (list, tuple)):
        boxes = []
        for b in raw:
            if _is_xyxy(b):
                boxes.append(list(map(float, b)))
    else:
        return []

    out = []
    for x1, y1, x2, y2 in boxes:
        if width is not None:
            x1 = max(0.0, min(x1, float(width)))
            x2 = max(0.0, min(x2, float(width)))
        if height is not None:
            y1 = max(0.0, min(y1, float(height)))
            y2 = max(0.0, min(y2, float(height)))
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _phrase_key(phrase: Any) -> str:
    if isinstance(phrase, list):
        return ' '.join(str(p) for p in phrase)
    return str(phrase or '').strip()


def _load_vg_items(path: str) -> List[dict]:
    """Load VG jsonlines or a JSON list."""
    with open(path, 'r') as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f'Expected a JSON list in {path}')
            return data
        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def _dump_vg_jsonlines(path: str, items: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def _offset_tokens(tokens_positive: Any, offset: int) -> List[List[int]]:
    if not tokens_positive:
        return []
    out = []
    for tok in tokens_positive:
        if (isinstance(tok, (list, tuple)) and len(tok) >= 2
                and isinstance(tok[0], (int, float))
                and isinstance(tok[1], (int, float))):
            out.append([int(tok[0]) + offset, int(tok[1]) + offset])
    return out


def merge_same_image_vg(items: List[dict],
                        keep_empty: bool = False) -> Tuple[List[dict], Counter]:
    """Merge VG items that share the same ``filename``.

    Logic mirrors ``_merge_raw_data_by_image`` in odvg_2_raw_data.py:
    captions joined with ``CAPTION_SEP``; categories (phrases) deduped in order;
    all valid bboxes kept under the merged caption with remapped tokens.
    """
    stats: Counter = Counter()
    stats['input_items'] = len(items)

    by_image: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        fn = item.get('filename', '')
        if not fn:
            stats['skip_empty_filename'] += 1
            continue
        by_image[fn].append(item)

    stats['unique_images'] = len(by_image)
    stats['multi_caption_images'] = sum(
        1 for v in by_image.values() if len(v) > 1)

    merged: List[dict] = []
    for filename, group in by_image.items():
        captions: List[str] = []
        # (caption_index, region_dict)
        pending: List[Tuple[int, dict]] = []
        width = group[0].get('width')
        height = group[0].get('height')

        for item in group:
            g = item.get('grounding') or {}
            cap = g.get('caption', '') or ''
            captions.append(cap)
            cap_idx = len(captions) - 1
            for region in g.get('regions', []) or []:
                pending.append((cap_idx, region))
            if width is None and 'width' in item:
                width = item['width']
            if height is None and 'height' in item:
                height = item['height']

        # Prefix length before caption i (including separators).
        offsets = []
        cur = 0
        for i, cap in enumerate(captions):
            offsets.append(cur)
            cur += len(cap)
            if i + 1 < len(captions):
                cur += len(CAPTION_SEP)
        merged_caption = CAPTION_SEP.join(captions)

        # Merge regions by phrase key (order = first appearance).
        regions_map: Dict[str, dict] = {}
        phrase_order: List[str] = []
        for cap_idx, region in pending:
            key = _phrase_key(region.get('phrase'))
            if not key:
                stats['drop_empty_phrase'] += 1
                continue
            boxes = _normalize_bboxes(
                region.get('bbox'), width=width, height=height)
            if not boxes:
                stats['drop_invalid_bbox'] += 1
                continue
            toks = _offset_tokens(
                region.get('tokens_positive'), offsets[cap_idx])
            if key not in regions_map:
                phrase_order.append(key)
                # Keep original phrase type (str or list) from first region.
                regions_map[key] = {
                    'bbox': boxes,
                    'phrase': region.get('phrase'),
                    'tokens_positive': toks,
                }
            else:
                regions_map[key]['bbox'].extend(boxes)
                # Append token spans from later captions for the same phrase.
                if toks:
                    regions_map[key]['tokens_positive'].extend(toks)

        region_list = [regions_map[k] for k in phrase_order]
        if not region_list and not keep_empty:
            stats['drop_empty_image'] += 1
            continue

        out = {
            'filename': filename,
            'grounding': {
                'caption': merged_caption,
                'regions': region_list,
            },
        }
        if width is not None:
            out['width'] = width
        if height is not None:
            out['height'] = height
        merged.append(out)
        stats['output_items'] += 1
        stats['output_regions'] += len(region_list)

    return merged, stats


def main():
    parser = argparse.ArgumentParser(
        description='Merge same-image Flickr30k/GQA ODVG VG annotations.')
    parser.add_argument(
        '--dataset',
        choices=sorted(DATASET_PRESETS.keys()),
        default=None,
        help='Optional preset for default input/output paths.')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default=None,
        help='Input ODVG VG json / jsonlines.')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output merged VG jsonlines.')
    parser.add_argument(
        '--keep-empty',
        action='store_true',
        help='Keep images that have no valid region after merge/filter.')
    args = parser.parse_args()

    if args.dataset:
        preset = DATASET_PRESETS[args.dataset]
        in_path = args.input or preset['default_input']
        out_path = args.output or preset['default_output']
    else:
        if not args.input or not args.output:
            parser.error(
                'Provide --input and --output, or choose --dataset flickr30k|gqa')
        in_path, out_path = args.input, args.output

    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    print(f'Loading {in_path} ...')
    items = _load_vg_items(in_path)
    print(f'  input lines: {len(items)}')

    merged, stats = merge_same_image_vg(items, keep_empty=args.keep_empty)
    _dump_vg_jsonlines(out_path, merged)

    print(f'Wrote {out_path}')
    print('Stats:')
    for k in sorted(stats):
        print(f'  {k}: {stats[k]}')


if __name__ == '__main__':
    main()
