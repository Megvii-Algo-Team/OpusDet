# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator

from mmdet.datasets.samplers.custom_sample_size_sampler import (
    CustomSampleSizeSampler as _Base)
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module(force=True)
class CustomSampleSizeSampler(_Base):
    """CustomSampleSizeSampler with fast-resume helpers for OPUS / TRex2 training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_samples = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        if self.start_samples > 0:
            indices = indices[self.start_samples:]
            self.start_samples = 0
        return iter(indices)

    def set_start_samples(self, n: int) -> None:
        """Skip first n samples on next ``__iter__``. For fast resume."""
        self.start_samples = int(n)

    def advance_cycle_for_epochs(self, n_epochs: int) -> None:
        """Advance cycle iters to simulate n_epochs. For fast resume."""
        for cycle_iter, data_size in zip(self.dataset_cycle_iter,
                                         self.dataset_size):
            if cycle_iter is not None and data_size != -1:
                for _ in range(n_epochs * data_size):
                    next(cycle_iter)
