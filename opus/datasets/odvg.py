# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path
from tqdm import tqdm

from mmdet.datasets.odvg import ODVGDataset as _ODVGDataset
from mmdet.registry import DATASETS


def _inter_area(x1, y1, x2, y2, data) -> float:
    if 'height' in data and 'width' in data:
        inter_w = max(0, min(x2, data['width']) - max(x1, 0))
        inter_h = max(0, min(y2, data['height']) - max(y1, 0))
    else:
        inter_w = max(0, x2 - max(x1, 0))
        inter_h = max(0, y2 - max(y1, 0))
    return inter_w * inter_h


@DATASETS.register_module(force=True)
class ODVGDataset(_ODVGDataset):
    """ODVG with tqdm, optional image size, and relaxed bbox clipping."""

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in tqdm(data_list, desc='load data list'):
            data_info = {
                'img_path': osp.join(self.data_prefix['img'], data['filename']),
            }
            if 'height' in data and 'width' in data:
                data_info['height'] = data['height']
                data_info['width'] = data['width']

            if self.dataset_mode == 'OD':
                out_data_list.append(
                    self._parse_od_record(data, data_info))
            else:
                out_data_list.append(
                    self._parse_vg_record(data, data_info))

        del data_list
        return out_data_list

    def _parse_od_record(self, data: dict, data_info: dict) -> dict:
        if self.need_text:
            data_info['text'] = self.label_map
        anno = data.get('detection', {})
        raw_instances = anno.get('instances', [])

        instances = []
        for obj in raw_instances:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            if _inter_area(x1, y1, x2, y2, data) == 0:
                continue
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            instances.append({
                'ignore_flag': 0,
                'bbox': bbox,
                'bbox_label': int(obj['label']),
            })

        data_info['instances'] = instances
        data_info['dataset_mode'] = self.dataset_mode
        return data_info

    def _parse_vg_record(self, data: dict, data_info: dict) -> dict:
        anno = data['grounding']
        data_info['text'] = anno['caption']

        instances = []
        phrases = {}
        for i, region in enumerate(anno['regions']):
            bbox = region['bbox']
            if not region.get('tokens_positive'):
                print(data, region)
            if not isinstance(bbox[0], list):
                bbox = [bbox]
            for box in bbox:
                x1, y1, x2, y2 = box
                if _inter_area(x1, y1, x2, y2, data) == 0:
                    continue
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                instances.append({
                    'ignore_flag': 0,
                    'bbox': box,
                    'bbox_label': i,
                })
                phrases[i] = {
                    'phrase': region['phrase'],
                    'tokens_positive': region['tokens_positive'],
                }

        data_info['instances'] = instances
        data_info['phrases'] = phrases
        data_info['dataset_mode'] = self.dataset_mode
        return data_info
