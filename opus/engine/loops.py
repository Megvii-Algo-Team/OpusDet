# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from mmengine.runner import IterBasedTrainLoop
from mmengine.runner.loops import _InfiniteDataloaderIterator
from mmengine.logging import print_log
from mmdet.registry import LOOPS

import os
DEBUG = os.getenv("DEBUG", '').lower() in ('y', 'yes', 'true', '1')


class _SequentialResumeSampler(Sampler):
    """Like ``SequentialSampler`` but with ``set_epoch`` / ``set_start_samples`` for resume.

    PyTorch's ``SequentialSampler`` has neither API, so fast resume falls back to
    slow ``next()``; this wrapper restores O(1) iterator reset after skip.
    """

    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.epoch = 0
        self.start_samples = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_samples(self, n: int) -> None:
        self.start_samples = int(n)

    def __iter__(self):
        n = len(self.data_source)
        start = min(max(0, self.start_samples), n)
        return iter(range(start, n))

    def __len__(self) -> int:
        n = len(self.data_source)
        return max(0, n - min(max(0, self.start_samples), n))


class AltIntervalScheduler:
    """
    Deterministic alternation scheduler based on
    base_alt_interval + alt_milestones.

    Semantics:
        interval = base_alt_interval
        for each milestone passed:
            interval /= alt_gamma

        interval >= 1:
            main : alt = interval : 1
        interval < 1:
            main : alt = 1 : (1 / interval)
    """

    def __init__(self, base_alt_interval, alt_milestones=None, alt_gamma: float = 1.0):
        self.base_alt_interval = float(base_alt_interval)
        self.alt_milestones = alt_milestones or []
        self.alt_gamma = float(alt_gamma)
        if self.alt_gamma <= 0:
            raise ValueError(f'alt_gamma must be > 0, got {self.alt_gamma}')

        self._pattern_cache = {}

    def _get_effective_interval(self, cur_iter: int) -> float:
        interval = self.base_alt_interval
        for m in self.alt_milestones:
            if cur_iter >= m:
                interval /= self.alt_gamma
            else:
                break
        return interval

    def _interval_to_ratio(self, interval: float):
        if interval >= 1.0:
            a = int(round(interval))
            b = 1
        else:
            a = 1
            b = int(round(1.0 / interval))
        return (a, b)

    def _get_pattern(self, ratio):
        if ratio not in self._pattern_cache:
            a, b = ratio
            # 0: main, 1: alt
            self._pattern_cache[ratio] = [0] * a + [1] * b
        return self._pattern_cache[ratio]

    def step(self, cur_iter: int):
        """
        Args:
            cur_iter (int): global iteration

        Returns:
            use_alt (bool)
            ratio (tuple): (main, alt)
        """
        interval = self._get_effective_interval(cur_iter)
        ratio = self._interval_to_ratio(interval)
        pattern = self._get_pattern(ratio)

        idx = cur_iter % len(pattern)
        use_alt = pattern[idx] == 1

        return use_alt, ratio
    
@LOOPS.register_module(force=True)
class OPUSIterBasedTrainLoop(IterBasedTrainLoop):
    """OPUS / TRex2 iter-based train loop with optional alt text/visual dataloader.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        resume_skip_data (bool): If True, advance dataloader to skip
            already-trained data on resume. Inside that path, ``_resume_via_sampler``
            (no ``next()`` loop) only works if the dataloader's sampler has
            ``set_epoch`` and ``set_start_samples`` (e.g. ``CustomSampleSizeSampler``).
            ``torch.utils.data.SequentialSampler`` is wrapped automatically.
            If False, **no** skip is performed (``_resume_via_sampler`` is never
            called); data order vs checkpoint may differ. Defaults to True.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        alt_gamma (float): After each passed ``alt_milestone``, the
            effective alt interval is divided by this factor (same as
            :class:`AltIntervalScheduler`). Defaults to ``1.0``.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            dataloader_alt: Union[DataLoader, Dict] = None,
            alt_mode:List[str] = ["text_visual",'text_only'],
            alt_interval: int = 8,
            alt_milestones=None,
            alt_gamma: float = 1.0,
            resume_skip_data: bool = True,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_iters,
            val_begin,
            val_interval,
            dynamic_intervals)

        # If False, resume without advancing dataloader (fast but data order differs)
        self.resume_skip_data = resume_skip_data
        # alt_interval schedule (historical ref:
        # https://github.com/IDEA-Research/T-Rex/issues/85#issuecomment-2422065213)
        # visual N iters, then text one iter;
        self.dataloader_alt_iterator = None
        self.alt_mode = None
        if dataloader_alt is not None:
            self.alt_mode = alt_mode
            self.base_alt_interval = alt_interval
            self.alt_milestones = alt_milestones
            self.alt_gamma = float(alt_gamma)
            self.alt_scheduler = AltIntervalScheduler(
                base_alt_interval=self.base_alt_interval,
                alt_milestones=self.alt_milestones,
                alt_gamma=self.alt_gamma,
            )
            if dataloader_alt == "same":
                self.dataloader_alt_iterator = self.dataloader_iterator
            else:
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                self.dataloader_alt = runner.build_dataloader(dataloader_alt, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                self.dataloader_alt_iterator = _InfiniteDataloaderIterator(self.dataloader_alt)
    
    def _resume_via_sampler(self, inf_iter, n_batches: int) -> bool:
        """Fast resume: set sampler epoch/start, recreate iterator. No data loading.
        Returns True if successful, False to fall back to _skip_iterator.

        Note: With AspectRatioBatchSampler, batch sizes can vary slightly; we use
        batch_size as an approximation. Prefer plain BatchSampler for exact resume.
        """
        def _fail(reason: str) -> bool:
            print_log(
                f'[resume] Fast path via sampler disabled: {reason}. '
                f'Using slow next() skip ({n_batches} steps). '
                f'For fast resume, use CustomSampleSizeSampler or sequential data '
                f'(SequentialSampler is wrapped automatically).',
                logger='current',
                level=logging.INFO)
            return False

        dataloader = getattr(inf_iter, '_dataloader', None)
        if dataloader is None:
            return _fail('no _dataloader on iterator')
        batch_sampler = getattr(dataloader, 'batch_sampler', None)
        # With custom batch_sampler, DataLoader.sampler is often an internal
        # SequentialSampler placeholder. The effective sampler is
        # batch_sampler.sampler and should be preferred for resume state.
        sampler = None
        if batch_sampler is not None:
            sampler = getattr(batch_sampler, 'sampler', None)
        if sampler is None:
            sampler = getattr(dataloader, 'sampler', None)
        if sampler is None:
            return _fail('no sampler (and batch_sampler has no .sampler)')
        if isinstance(sampler, SequentialSampler):
            wrapped = _SequentialResumeSampler(sampler.data_source)
            if batch_sampler is not None:
                batch_sampler.sampler = wrapped
            else:
                dataloader.sampler = wrapped
            sampler = wrapped
        if not hasattr(sampler, 'set_epoch'):
            return _fail(f'sampler {type(sampler).__name__} has no set_epoch')
        if not hasattr(sampler, 'set_start_samples'):
            return _fail(f'sampler {type(sampler).__name__} has no set_start_samples')

        # DataLoader.batch_size is None when using custom batch_sampler
        batch_size = getattr(dataloader, 'batch_size', None)
        if batch_size is None and batch_sampler is not None:
            batch_size = getattr(batch_sampler, 'batch_size', 1)
        batch_size = batch_size or 1
        if not isinstance(batch_size, int):
            return _fail(
                f'non-int batch_size={batch_size!r} (e.g. multi-size batch sampler) '
                f'is not supported for sample-based skip')

        # len(dataloader) = len(_index_sampler) = len(batch_sampler) when using batch_sampler
        batches_per_epoch = len(dataloader)
        if batches_per_epoch <= 0:
            return _fail('len(dataloader)==0')

        epoch = n_batches // batches_per_epoch
        start_batch_in_epoch = n_batches % batches_per_epoch
        start_samples = start_batch_in_epoch * batch_size

        # Advance cycle iters for cross-epoch resume (CustomSampleSizeSampler)
        if hasattr(sampler, 'advance_cycle_for_epochs'):
            sampler.advance_cycle_for_epochs(epoch)

        sampler.set_epoch(epoch)
        sampler.set_start_samples(start_samples)
        inf_iter._iterator = iter(inf_iter._dataloader)

        print_log(
            f'Fast resume via sampler: skipped {n_batches} iters '
            f'(epoch={epoch}, iters_in_epoch={start_batch_in_epoch}, '
            f'batches_per_epoch={batches_per_epoch})',
            logger='current',
            level=logging.INFO)
        return True

    def _skip_iterator(self, inf_iter, n: int, pbar=None) -> None:
        """Advance iterator by n steps. Tries sampler-based fast path first."""
        if n <= 0:
            return
        if self._resume_via_sampler(inf_iter, n):
            if pbar is not None:
                pbar.update(n)
            return
        # Slow path: compute epoch/iters for logging
        dataloader = getattr(inf_iter, '_dataloader', None)
        batches_per_epoch = len(dataloader) if dataloader else 0
        epoch_skip = n // batches_per_epoch if batches_per_epoch > 0 else 0
        iters_in_epoch = n % batches_per_epoch if batches_per_epoch > 0 else n
        print_log(
            f'Resume via next(): skipping {n} iters '
            f'(epoch={epoch_skip}, iters_in_epoch={iters_in_epoch}, '
            f'batches_per_epoch={batches_per_epoch})',
            logger='current',
            level=logging.INFO)
        for _ in range(n):
            next(inf_iter)
            if pbar is not None:
                pbar.update(1)

    def _get_data_batch_and_mode(self) -> Tuple[Union[dict, tuple], Optional[str]]:
        """Get next data_batch and mode. Handles main/alt alternation."""
        if self.alt_mode is not None:
            use_alt, ratio = self.alt_scheduler.step(self._iter)
            if use_alt:
                data_batch = next(self.dataloader_alt_iterator)
                mode = self.alt_mode[1]
            else:
                data_batch = next(self.dataloader_iterator)
                mode = self.alt_mode[0]
            data_batch['data_samples'][0].set_metainfo({"mode": mode})
            if DEBUG:
                print(
                    f"[iter {self._iter}] ratio={ratio} mode={mode}"
                )
            return data_batch, mode
        else:
            return next(self.dataloader_iterator), None

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')

        # Resume: advance dataloader to skip already-trained data (or skip for fast resume)
        if self._iter > 0:
            if self.resume_skip_data:
                print_log(
                    f'Advance dataloader {self._iter} steps to skip data '
                    'that has already been trained',
                    logger='current',
                    level=logging.WARNING)
                if self.alt_mode is not None:
                    n_main = sum(1 for i in range(self._iter)
                                if not self.alt_scheduler.step(i)[0])
                    n_alt = self._iter - n_main
                    if self.dataloader_alt_iterator is not self.dataloader_iterator:
                        with tqdm(total=n_main + n_alt, desc="skip data") as pbar:
                            self._skip_iterator(
                                self.dataloader_iterator, n_main, pbar=pbar)
                            self._skip_iterator(
                                self.dataloader_alt_iterator, n_alt, pbar=pbar)
                    else:
                        with tqdm(total=self._iter, desc="skip data") as pbar:
                            self._skip_iterator(
                                self.dataloader_iterator, self._iter, pbar=pbar)
                else:
                    with tqdm(total=self._iter, desc="skip data") as pbar:
                        self._skip_iterator(
                            self.dataloader_iterator, self._iter, pbar=pbar)
            else:
                print_log(
                    'Fast resume: dataloader not advanced (data order may differ)',
                    logger='current',
                    level=logging.INFO)

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch, mode = self._get_data_batch_and_mode()
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Union[dict, tuple]) -> None:
        """Run one training iteration with the given data_batch."""
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
