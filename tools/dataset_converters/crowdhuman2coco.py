#!/usr/bin/env python3
"""Convert CrowdHuman ODGT annotations to COCO JSON.

Adapted from MMDetection ``tools/dataset_converters/crowdhuman2coco.py``.

Expects under ``--input``::

    annotation_train.odgt
    annotation_val.odgt
    train/Images/{ID}.jpg
    val/Images/{ID}.jpg

Writes ``crowdhuman_train.json`` / ``crowdhuman_val.json`` under ``--output``,
with ``file_name`` as ``Images/{ID}.jpg`` (use a symlink
``data/crowdhuman/Images -> train/Images`` for training).

Example::

  python tools/dataset_converters/crowdhuman2coco.py \\
    -i data/crowdhuman \\
    -o data/crowdhuman/annotations
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import mmengine
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='CrowdHuman ODGT → COCO JSON (OPUS)')
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='root directory of CrowdHuman (odgt + train/ val/ Images)',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='directory to save coco formatted label files',
    )
    return parser.parse_args()


def load_odgt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [json.loads(line.strip('\n')) for line in lines]


def convert_crowdhuman(ann_dir, save_dir, mode='train'):
    """Convert CrowdHuman dataset in COCO style.

    Args:
        ann_dir (str): The path of CrowdHuman dataset.
        save_dir (str): The path to save annotation files.
        mode (str): Convert train or val. Default: 'train'.
    """
    assert mode in ['train', 'val']

    records = dict(img_id=1, ann_id=1)
    outputs = defaultdict(list)
    outputs['categories'] = [dict(id=1, name='pedestrian')]

    data_infos = load_odgt(osp.join(ann_dir, f'annotation_{mode}.odgt'))
    for data_info in tqdm(data_infos, desc=f'CrowdHuman {mode}'):
        img_name = osp.join('Images', f"{data_info['ID']}.jpg")
        img = Image.open(osp.join(ann_dir, mode, img_name))
        width, height = img.size[:2]
        image = dict(
            file_name=img_name,
            height=height,
            width=width,
            id=records['img_id'])
        outputs['images'].append(image)

        for ann_info in data_info['gtboxes']:
            bbox = ann_info['fbox']
            if ('extra' in ann_info and 'ignore' in ann_info['extra']
                    and ann_info['extra']['ignore'] == 1):
                iscrowd = True
            else:
                iscrowd = False
            ann = dict(
                id=records['ann_id'],
                image_id=records['img_id'],
                category_id=outputs['categories'][0]['id'],
                vis_bbox=ann_info['vbox'],
                bbox=bbox,
                area=bbox[2] * bbox[3],
                iscrowd=iscrowd)
            outputs['annotations'].append(ann)
            records['ann_id'] += 1
        records['img_id'] += 1

    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, f'crowdhuman_{mode}.json')
    mmengine.dump(outputs, out_path)
    print(f'-----CrowdHuman {mode} set------')
    print(f'total {records["img_id"] - 1} images')
    print(f'{records["ann_id"] - 1} pedestrians are annotated.')
    print(f'save to {out_path}')
    print('-----------------------')


def main():
    args = parse_args()
    convert_crowdhuman(args.input, args.output, mode='train')
    convert_crowdhuman(args.input, args.output, mode='val')


if __name__ == '__main__':
    main()
