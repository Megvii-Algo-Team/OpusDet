#!/usr/bin/env python3
"""Convert HierText hierarchical annotations to COCO JSON (word-level).

Adapted from ``tools/dataset_converters/hiertext2coco.py``.

Official HierText ``train.jsonl`` / ``validation.jsonl`` is **not** COCO: each
image has nested ``paragraphs → lines → words`` with polygon ``vertices``.
This script loads that JSON object and emits word-level COCO boxes
(``category`` = ``text``), skipping illegible lines/words.

Example::

  # after gzip -d annotations/train.jsonl.gz
  python tools/dataset_converters/hiertext2coco.py \\
    -i data/hiertext/annotations/train.jsonl \\
    -o data/hiertext/annotations/train.json
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='HierText hierarchical JSON → COCO JSON (word-level)')
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='HierText annotation file (JSON object with "annotations")',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='output COCO JSON path (e.g. annotations/train.json)',
    )
    return parser.parse_args()


def polygon_to_bbox(polygon):
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x_min = min(xs)
    y_min = min(ys)
    w = max(xs) - x_min
    h = max(ys) - y_min
    return [x_min, y_min, w, h]


def convert_hiertext_to_coco(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        hier_data = json.load(f)

    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'text'}]
    ann_id = 1

    for img_id, item in enumerate(
            tqdm(hier_data['annotations'], desc='Converting')):
        file_name = item['image_id'] + '.jpg'
        width = item['image_width']
        height = item['image_height']

        images.append({
            'id': img_id,
            'file_name': file_name,
            'width': width,
            'height': height,
        })

        for para in item['paragraphs']:
            for line in para['lines']:
                if not line.get('legible', True):
                    continue
                for word in line['words']:
                    vertices = word['vertices']
                    if not word.get('legible', True) or not vertices:
                        continue
                    bbox = polygon_to_bbox(vertices)
                    segmentation = [
                        coord for point in vertices for coord in point
                    ]
                    area = bbox[2] * bbox[3]
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': bbox,
                        'area': area,
                        'iscrowd': 0,
                        'segmentation': [segmentation],
                    })
                    ann_id += 1

    coco_output = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

    print(f'Converted {len(images)} images, {len(annotations)} words')
    print(f'save to {output_path}')


def main():
    args = parse_args()
    convert_hiertext_to_coco(args.input, args.output)


if __name__ == '__main__':
    main()
