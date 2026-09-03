# Copyright (c) OpenMMLab. All rights reserved.
"""Wall-clock segment timers for OPUS speed benchmark
(``tools/analysis_tools/get_flops.py --speed-test``).
"""
from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager
from typing import Dict, FrozenSet, Optional, Tuple

import torch

# Segment groups for derived latency / FPS helpers (see OPUSSpeedProfiler.finalize).
INTERACTIVE_KEYS: Tuple[str, ...] = (
    'vpg_prepare',
    'vpg_roi_align',
    'visual_prompt_generator',
    'vpg',
    'pre_decoder',
    'decoder',
    'head',
)
PROMPT_ENCODING_KEYS: FrozenSet[str] = frozenset({
    'text_encoder',
    'misc_text_len_check',
    'prompt_prep',
    'vpg_prepare',
    'vpg_roi_align',
    'visual_prompt_generator',
    'vpg',
})
SKIP_SUM_KEYS: FrozenSet[str] = frozenset({
    'forward_wall_ms',
    'gap_unaccounted_ms',
    'e2e_ms',
    'cache_prompt_ms',
    'interactive_ms',
})


def opus_segment(profiler: Optional['OPUSSpeedProfiler'], name: str):
    """No-op context when profiler is None or segments disabled."""
    if profiler is not None and profiler.enabled and profiler.profile_segments:
        return profiler.segment(name)
    return contextlib.nullcontext()


class OPUSSpeedProfiler:
    """Per-forward segment timings (ms), compatible with get_flops.py speed_test."""

    def __init__(self,
                 enabled: bool = True,
                 profile_segments: bool = True) -> None:
        self.enabled = bool(enabled)
        self.profile_segments = bool(profile_segments)
        self._use_cuda = torch.cuda.is_available()
        self._records: Dict[str, float] = {}
        self._wall_start: Optional[float] = None

    def start_wall(self) -> None:
        if self.enabled:
            self._wall_start = time.perf_counter()

    @contextmanager
    def segment(self, name: str):
        if not self.enabled or not self.profile_segments:
            yield
            return
        if self._use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._use_cuda:
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            self._records[name] = self._records.get(name, 0.0) + dt

    def finalize(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        stats: Dict[str, float] = {}
        if self._wall_start is not None:
            wall_ms = (time.perf_counter() - self._wall_start) * 1000.0
            stats['forward_wall_ms'] = wall_ms
            stats['e2e_ms'] = wall_ms
        if self.profile_segments:
            stats.update(self._records)
            seg_sum = sum(
                v for k, v in stats.items() if k not in SKIP_SUM_KEYS)
            wall = stats.get('forward_wall_ms')
            if wall is not None:
                stats['gap_unaccounted_ms'] = max(0.0, wall - seg_sum)
            self._add_derived(stats)
        elif stats.get('forward_wall_ms') is not None:
            stats['gap_unaccounted_ms'] = 0.0
        return stats

    @staticmethod
    def _add_derived(stats: Dict[str, float]) -> None:
        cache_ms = sum(
            stats.get(k, 0.0) for k in stats
            if k not in SKIP_SUM_KEYS and k not in PROMPT_ENCODING_KEYS)
        if cache_ms > 0:
            stats['cache_prompt_ms'] = cache_ms

        interactive_ms = sum(stats.get(k, 0.0) for k in INTERACTIVE_KEYS)
        if interactive_ms > 0:
            stats['interactive_ms'] = interactive_ms


# Back-compat aliases for older TRex2 call sites / analysis scripts.
TreX2SpeedProfiler = OPUSSpeedProfiler
trex2_segment = opus_segment
