# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

from mmengine.dataset import BaseDataset

from mmdet.datasets.dataset_wrappers import ConcatDataset as _ConcatDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(force=True)
class ConcatDataset(_ConcatDataset):
    """ConcatDataset: stop comparing metainfo once a mismatch is found."""

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[BaseDataset] = []
        for dataset in datasets:
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()

        is_all_same = True
        self._metainfo_first = self.datasets[0].metainfo
        for dataset in self.datasets[1:]:
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    is_all_same = False
                    break
                if self._metainfo_first[key] != dataset.metainfo[key]:
                    is_all_same = False
                    break
            if not is_all_same:
                break

        if is_all_same:
            self._metainfo = self.datasets[0].metainfo
        else:
            self._metainfo = [dataset.metainfo for dataset in self.datasets]

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()
            if is_all_same:
                self._metainfo.update(
                    dict(cumulative_sizes=self.cumulative_sizes))
            else:
                for i, dataset in enumerate(self.datasets):
                    self._metainfo[i].update(
                        dict(cumulative_sizes=self.cumulative_sizes))
