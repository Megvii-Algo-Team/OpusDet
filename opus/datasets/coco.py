# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

from mmengine.fileio import get_local_path

from mmdet.datasets.coco import CocoDataset as _CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(force=True)
class CocoDataset(_CocoDataset):
    """COCO dataset with class names synced from annotation categories."""

    def _sync_classes_from_ann(self) -> None:
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self._metainfo['classes'] = tuple(
            c['name'] for c in self.coco.load_cats(self.cat_ids))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        self._sync_classes_from_ann()

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            data_list.append(self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info,
            }))

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list
