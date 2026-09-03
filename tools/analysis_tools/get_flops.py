# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import inspect
import os
import re
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import digit_version

from mmdet.registry import MODELS
from opus.utils.strip_pretrained import strip_pretrained_for_flops

try:
    from opus.utils.speed_profiler import SKIP_SUM_KEYS
except ImportError:
    SKIP_SUM_KEYS = frozenset({
        'forward_wall_ms',
        'gap_unaccounted_ms',
        'e2e_ms',
        'cache_prompt_ms',
        'interactive_ms',
    })

try:
    from mmengine.analysis import ActivationAnalyzer, FlopAnalyzer, parameter_count
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def _robust_jit_op_handles(analyzer) -> None:
    """Wrap FlopAnalyzer / ActivationAnalyzer op counters so JIT shape edge cases
    do not abort the trace (mmengine jit_handlers often assume complete shapes and
    use inputs[0]/outputs[0]; OPUS + transformers can hit IndexError there).
    """
    try:
        handles = list(analyzer._op_handles.items())
    except AttributeError:
        return
    for op_name, handle in handles:
        if handle is None:
            continue

        def _wrap(h):
            def _safe(inputs, outputs):
                try:
                    return h(inputs, outputs)
                except (
                    IndexError,
                    TypeError,
                    AssertionError,
                    AttributeError,
                    NotImplementedError,
                    ValueError,
                ):
                    return 0

            return _safe

        analyzer.set_op_handle(**{op_name: _wrap(handle)})


def _install_robust_flop_analyzer_hooks() -> None:
    """Patch FlopAnalyzer / ActivationAnalyzer so get_model_complexity_info() uses
    the same analyzers but with robust per-op handlers (see _robust_jit_op_handles).
    """
    if getattr(FlopAnalyzer, '_get_flops_robust_wrapped', False):
        return
    _orig_flop = FlopAnalyzer.__init__
    _orig_act = ActivationAnalyzer.__init__

    def _flop_init(self, model, inputs):
        _orig_flop(self, model, inputs)
        _robust_jit_op_handles(self)

    def _act_init(self, model, inputs):
        _orig_act(self, model, inputs)
        _robust_jit_op_handles(self)

    FlopAnalyzer.__init__ = _flop_init  # type: ignore[assignment]
    ActivationAnalyzer.__init__ = _act_init  # type: ignore[assignment]
    FlopAnalyzer._get_flops_robust_wrapped = True


_install_robust_flop_analyzer_hooks()


def _install_jit_analyze_per_node_guard() -> None:
    """Patch mmengine JitModelAnalysis._analyze: some PyTorch/transformer traces
    produce JIT nodes where scope handling or op counters raise IndexError/KeyError.
    Skip those nodes instead of aborting the whole FlopAnalyzer run.
    """
    import warnings
    from collections import Counter
    from numbers import Number

    import numpy as np
    from torch.jit import TracerWarning

    from mmengine.analysis import jit_analysis as ja
    from mmengine.analysis.jit_analysis import (
        Statistics,
        _named_modules_with_dup,
    )

    if getattr(ja.JitModelAnalysis, '_get_flops_analyze_patched', False):
        return

    def _patched_analyze(self):
        stats = self._stats
        if stats is not None:
            return stats

        with warnings.catch_warnings():
            if self._warn_trace == 'none':
                warnings.simplefilter('ignore')
            elif self._warn_trace == 'no_tracer_warning':
                warnings.filterwarnings('ignore', category=TracerWarning)
            graph = ja._get_scoped_trace_graph(
                self._model, self._inputs, self._aliases)

        counts = {}
        unsupported_ops = {}
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()

        all_seen = set()
        for node in graph.nodes():
            try:
                kind = node.kind()
                if kind == 'prim::PythonOp':
                    kind = kind + '.' + node.pyname()
                scope_names = node.scopeName().split('/')
                all_seen.update(scope_names)
                if self._ancestor_mode == 'caller':
                    ancestors = set(scope_names)
                else:
                    if not scope_names:
                        continue
                    ancestors = self._get_all_ancestors(scope_names[-1])
                    all_seen.update(ancestors)
                if kind not in self._op_handles:
                    if self._should_ignore_node(node):
                        continue
                    for name in ancestors:
                        unsupported_ops[name][kind] += 1
                else:
                    inputs, outputs = list(node.inputs()), list(node.outputs())
                    op_counts = self._op_handles[kind](inputs, outputs)
                    if isinstance(op_counts, Number):
                        op_counts = Counter(
                            {self._simplify_op_name(kind): op_counts})
                    for v in op_counts.values():
                        if not isinstance(v, (int, float, np.float64, np.int64)):
                            raise ValueError(
                                f'Invalid type {type(v)} for the flop count! '
                                'Please use a wider type to avoid overflow.')
                    for name in ancestors:
                        counts[name] += op_counts
            except (IndexError, KeyError, AttributeError):
                continue

        uncalled_mods = set(self._aliases.values()) - all_seen
        stats = Statistics(
            counts=counts,
            unsupported_ops=unsupported_ops,
            uncalled_mods=uncalled_mods)
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[''])
        self._warn_uncalled_mods(uncalled_mods)
        return stats

    ja.JitModelAnalysis._analyze = _patched_analyze
    ja.JitModelAnalysis._get_flops_analyze_patched = True


_install_jit_analyze_per_node_guard()


def _install_complexity_stats_table_safe() -> None:
    """get_model_complexity_info() calls complexity_stats_table() after total();
    deep/transformer models can raise IndexError inside table formatting (not in
    JitModelAnalysis._analyze). Patch print_helper so totals still return."""
    import mmengine.analysis.print_helper as ph

    if getattr(ph, '_get_flops_cst_safe', False):
        return
    _orig = ph.complexity_stats_table

    def _safe(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except (IndexError, KeyError):
            return (
                '\n(FlopAnalyzer per-layer table failed to render; '
                'totals from total() are still valid.)\n')

    ph.complexity_stats_table = _safe
    ph._get_flops_cst_safe = True


_install_complexity_stats_table_safe()


def _flops_failure_hint(exc: Exception) -> str:
    """Short explanation when FlopAnalyzer JIT trace fails (often HF CLIP + transformers)."""
    try:
        tb = ''.join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb = ''
    if any(
            s in tb
            for s in ('masking_utils', 'create_causal_mask', 'sdpa_mask')):
        return (
            ' (transformers causal/SDPA mask code is not robust under torch.jit '
            'trace; q_length indexing fails. Not fixable in mmengine — use '
            '--skip-flops for timing-only, or cite params / other estimates for FLOPs.)')
    if 'modeling_clip' in tb:
        return (
            ' (CLIP text_model forward failed under JIT trace; use --skip-flops '
            'for timing-only runs.)')
    return ''


def _shape_hw_str(shape) -> str:
    """Format spatial size as (H, W), matching MMDet ori_shape / batch_input_shape."""
    if shape is None:
        return str(shape)
    try:
        h, w = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        return str(shape)
    return f'(H,W)=({h}, {w})'


def _fmt_table(headers: list, rows: list) -> str:
    """Fixed-width pipe table (compact console / file output)."""
    if not rows:
        return ''
    n = len(headers)
    cells = [headers] + [list(r) for r in rows]
    widths = [0] * n
    for r in cells:
        for i in range(n):
            widths[i] = max(widths[i], len(str(r[i])))
    sep = '-+-'.join('-' * w for w in widths)

    def _line(r):
        return '| ' + ' | '.join(
            str(r[i]).ljust(widths[i]) for i in range(n)) + ' |'

    return '\n'.join([_line(headers), sep] + [_line(r) for r in rows])


def _build_depth1_params_table(model: torch.nn.Module) -> str:
    """Non-overlapping param counts: each row is one direct child's full subtree."""
    total = sum(p.numel() for p in model.parameters())
    if total == 0:
        return ''
    rows = []
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n == 0:
            continue
        pct = 100.0 * float(n) / float(total)
        rows.append((name, n, pct))
    rows.sort(key=lambda x: -x[1])
    table_rows = [[
        name,
        _format_size(n),
        f'{pct:.2f}%',
    ] for name, n, pct in rows]
    if not table_rows:
        return _fmt_table(
            ['module (depth-1)', 'params', '% of total'],
            [['(all parameters on root)', _format_size(total), '100.00%']])
    table_rows.append(['TOTAL', _format_size(total), '100.00%'])
    return _fmt_table(['module (depth-1)', 'params', '% of total'], table_rows)


def _parse_img_size(values):
    """Return ``(h, w)`` for pipeline resize/pad from CLI ``--img-size`` (same as ``tools/test.py``)."""
    if values is None:
        return None
    if len(values) == 1:
        s = int(values[0])
        return (s, s)
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError(
        '--img-size expects one int (square) or two ints (height width)')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=20,
        help='num images of calculate model flops')
    parser.add_argument(
        '--show-table',
        dest='show_table',
        action='store_true',
        help='include FlopAnalyzer per-layer FLOPs table (can be large)')
    parser.add_argument(
        '--no-show-table',
        dest='show_table',
        action='store_false',
        help='omit FlopAnalyzer per-layer table to shorten the report')
    parser.set_defaults(show_table=True)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--speed-test',
        action='store_true',
        default=False,
        help='whether to run inference speed benchmark')
    parser.add_argument(
        '--warmup-iter',
        type=int,
        default=20,
        help='number of warmup iterations for speed benchmark')
    parser.add_argument(
        '--speed-iter',
        type=int,
        default=100,
        help='number of measured iterations for speed benchmark')
    parser.add_argument(
        '--measure-mode',
        type=str,
        default='tensor',
        choices=['tensor', 'predict'],
        help='benchmark mode: tensor for raw forward, predict for full inference')
    parser.add_argument(
        '--no-segment-timing',
        action='store_true',
        default=False,
        help='OPUS: E2E wall clock only; skips segment table and '
        'fps_cache_prompt / fps_interactive derived timings')
    parser.add_argument(
        '--prompt-mode',
        type=str,
        default='text_visual.I.1', # ['text_only', 'visual.G', 'text_visual.I.1'],
        help='prompt mode for model forward speed test')
    parser.add_argument(
        '--speed-num-classes',
        type=int,
        default=0,
        help='OPUS speed: keep at most N entity classes (trim gt_instances + '
        'text). 0 = all classes in sample (default).')
    parser.add_argument(
        '--img-size',
        type=int,
        nargs='+',
        default=None,
        metavar=('H', 'W'),
        help='target tensor spatial size; patches pipeline ``FixScaleResize`` / ``Resize`` '
        '``scale`` and ``Pad`` ``size`` when present. One int: square ``(s, s)``; '
        'two ints: ``(height, width)`` (same order as ``tools/test.py`` ``--img-size``).')
    parser.add_argument(
        '--out-prefix',
        type=str,
        default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--out-file',
        type=str,
        default=None,
        help='output directory for report '
        '(default: work_dirs/flops_speed/{config_name})')
    parser.add_argument(
        '--switch-to-deploy',
        action='store_true',
        default=False,
        help='call switch_to_deploy() on submodules that define it (inference reparameterization)')
    parser.add_argument(
        '--skip-flops',
        action='store_true',
        default=False,
        help='skip mmengine FlopAnalyzer (HF CLIP text_model + transformers masking_utils '
            'often raise IndexError under JIT trace — not an mmengine bug); '
            'still runs forward for shapes; use with --speed-test for timing-only')
    parser.add_argument(
        '--verbose-report',
        action='store_true',
        default=False,
        help='long text: interpretation block, latency source notes, module p50/p90/p95 columns')
    parser.add_argument(
        '--decoder-early-exit-layer',
        type=int,
        default=None,
        help='OPUS inference only: set model.test_cfg.decoder_early_exit_layer '
        '(must be > 0)')
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='checkpoint path (overrides config ``load_from``); loaded after '
        'MODELS.build with HF pretrain stripped')
    args = parser.parse_args()
    if args.img_size is not None:
        try:
            args.img_size = _parse_img_size(args.img_size)
        except ValueError as e:
            parser.error(str(e))
    return args


def _safe_filename_segment(s: str) -> str:
    s = str(s).replace('/', '_').replace('\\', '_')
    s = re.sub(r'[^0-9A-Za-z._-]+', '_', s)
    return s.strip('_') or 'x'


def _default_work_dir_from_config(config_path: str) -> Path:
    """Default FLOPs/speed work dir: work_dirs/flops_speed/{config_parent_dir}."""
    p = Path(config_path)
    parent_dir = _safe_filename_segment(p.parent.name)
    return Path('work_dirs') / 'flops_speed' / parent_dir


def build_auto_output_stem(args: argparse.Namespace) -> str:
    """Filename stem (no extension) from config + profiling-related CLI flags."""
    parts = [_safe_filename_segment(Path(args.config).stem)]
    if args.img_size is not None:
        parts.append(f'{int(args.img_size[0])}x{int(args.img_size[1])}')
    if args.skip_flops:
        parts.append('noflops')
    if getattr(args, 'switch_to_deploy', False):
        parts.append('deploy')
    parts.append(f'ni{args.num_images}')
    if args.show_table:
        parts.append('flopstbl')
    if args.speed_test:
        parts.append(_safe_filename_segment(args.prompt_mode))
        parts.append(args.measure_mode)
        if getattr(args, 'decoder_early_exit_layer', None) is not None:
            parts.append(f'decx{int(args.decoder_early_exit_layer)}')
        if getattr(args, 'no_segment_timing', False):
            parts.append('noseg')
        else:
            parts.append('seg')
        parts.append(f'w{args.warmup_iter}')
        parts.append(f's{args.speed_iter}')
        snc = int(getattr(args, 'speed_num_classes', 0) or 0)
        if snc > 0:
            parts.append(f'snc{snc}')
    return '__'.join(parts)


def resolve_output_paths(args: argparse.Namespace, logger: MMLogger | None = None) -> None:
    """Resolve output directory and derived output file paths.

    Unified behavior:
    - ``--out-file`` is treated as output directory.
    - report path is auto-generated as ``{stem}_report.txt`` under that directory.
    - ``--out-prefix`` is deprecated; if provided and ``--out-file`` keeps default,
      it is used as output directory for backward compatibility.
    """
    default_out_dir = _default_work_dir_from_config(args.config)
    out_dir_arg = args.out_file
    if args.out_prefix:
        if out_dir_arg is None:
            out_dir_arg = args.out_prefix
        if logger is not None:
            logger.warning('--out-prefix is deprecated; use --out-file <output_dir>.')
    if out_dir_arg is None:
        out_dir_arg = str(default_out_dir)

    out_dir = Path(out_dir_arg).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = build_auto_output_stem(args)
    args.out_dir = str(out_dir)
    args.report_out_file = str(out_dir / f'{stem}_report.txt')


def apply_inference_deploy_opts(model, args, logger):
    """Optional Conv+BN fuse and switch_to_deploy before profiling."""
    if getattr(args, 'switch_to_deploy', False):
        if hasattr(model, 'deploy'):
            model = model.deploy()
            logger.info('Applied model.deploy() for inference profiling.')
        else:
            n = 0
            for m in model.modules():
                if hasattr(m, 'convert_to_deploy'):
                    m.convert_to_deploy()
                    n += 1
            logger.info('switch_to_deploy fallback: %d module(s).', n)
    return model


def set_keys_to_zero(cfg, keys=('with_cp', 'num_cp')):
    def recursive_set(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    obj[k] = 0
                else:
                    recursive_set(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_set(item)
        elif hasattr(obj, '__dict__'):
            recursive_set(vars(obj))

    recursive_set(cfg)
    return cfg


def _resolve_load_from(cfg, args) -> Optional[str]:
    """CLI ``--load-from`` overrides config ``load_from``."""
    cli = getattr(args, 'load_from', None)
    if cli is not None and str(cli).strip():
        return str(cli).strip()
    ckpt = cfg.get('load_from', None)
    if ckpt is not None and str(ckpt).strip():
        return str(ckpt).strip()
    return None


def _load_model_checkpoint(model, load_from: str, logger=None) -> None:
    path = Path(load_from).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'load_from checkpoint not found: {path}')
    if logger is not None:
        logger.info('Loading checkpoint: %s', path)
    load_checkpoint(model, str(path), map_location='cpu')


def _enable_last_layer_head_only_in_cfg(cfg, logger=None):
    """Enable ``inference_head_last_layer_only`` in ``cfg.test_cfg``."""
    model_cfg = cfg.setdefault('model', {})
    test_cfg = model_cfg.get('test_cfg', None)
    if not isinstance(test_cfg, Mapping):
        model_cfg['test_cfg'] = dict()
    model_cfg['test_cfg']['inference_head_last_layer_only'] = True
    logger.info(
        'Set model.test_cfg.inference_head_last_layer_only=True')


def _cfg_mapping_root(cfg_obj):
    """mmengine ``Config`` is not a ``dict``; it stores content in ``_cfg_dict``.

    Nested nodes are often ``ConfigDict`` (addict ``Dict``), which also does not
    satisfy ``isinstance(..., dict)`` in some versions. Always unwrap the top
    ``Config`` so traversal can find ``val_dataloader.dataset.pipeline``.
    """
    if hasattr(cfg_obj, '_cfg_dict'):
        return cfg_obj._cfg_dict
    return cfg_obj


def _is_mapping_node(obj):
    return isinstance(obj, Mapping) and not isinstance(obj, (str, bytes))


def _iter_pipelines(cfg_obj):
    """Yield (parent_dict, key, pipeline_list) for any dict field named
    `pipeline` that contains a list.

    We do this recursively because mmdet configs often define dataset/pipeline
    inside nested base datasets.
    """
    cfg_obj = _cfg_mapping_root(cfg_obj)
    if _is_mapping_node(cfg_obj):
        for k, v in cfg_obj.items():
            if k == 'pipeline' and isinstance(v, (list, tuple)):
                yield cfg_obj, k, v
            else:
                yield from _iter_pipelines(v)
    elif isinstance(cfg_obj, (list, tuple)):
        for it in cfg_obj:
            yield from _iter_pipelines(it)


def _patch_resize_and_pad_transforms(pipeline, h: int, w: int) -> int:
    """Patch ``FixScaleResize.scale`` inside a pipeline list.

    Sets ``scale`` to ``(min(H,W), max(H,W))`` — the (short_edge, long_edge) pair
    used by :class:`FixScaleResize` with ``keep_ratio=True`` / :func:`imrescale`.
    Does not modify ``keep_ratio`` on the transform (leave config as-is).

    Args:
        h, w: Target tensor height and width (``--img-size H W``).
    """
    patched = 0
    for t in pipeline:
        if not isinstance(t, dict):
            continue
        t_type = t.get('type', None)

        if t_type == 'FixScaleResize' and 'scale' in t:
            t['scale'] = (min(h, w), max(h, w))
            patched += 1
        if t_type == 'Pad' and 'size' in t:
            t['size'] = (w, h)
            patched += 1
        # Recursively patch nested transforms (e.g. RandomChoice -> transforms=[...])
        if isinstance(t.get('transforms', None), list):
            nested = t['transforms']
            for item in nested:
                if isinstance(item, list):
                    patched += _patch_resize_and_pad_transforms(item, h=h, w=w)
                elif isinstance(item, dict):
                    patched += _patch_resize_and_pad_transforms([item], h=h, w=w)

    return patched

def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg = set_keys_to_zero(cfg)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if getattr(args, 'load_from', None):
        cfg.load_from = args.load_from
    if args.decoder_early_exit_layer is not None:
        # if int(args.decoder_early_exit_layer) <= 0:
        #     raise ValueError('--decoder-early-exit-layer must be > 0')
        model_cfg = cfg.setdefault('model', {})
        test_cfg = model_cfg.get('test_cfg', None)
        if not isinstance(test_cfg, Mapping):
            model_cfg['test_cfg'] = dict()
        model_cfg['test_cfg']['decoder_early_exit_layer'] = int(
            args.decoder_early_exit_layer)
        logger.info(
            'Set model.test_cfg.decoder_early_exit_layer=%d',
            int(args.decoder_early_exit_layer))

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Optional: override the input shape by patching dataset pipeline transforms.
    if args.img_size is not None:
        target_h, target_w = int(args.img_size[0]), int(args.img_size[1])
        patched_total = 0
        for parent, key, pipeline in _iter_pipelines(cfg):
            patched_total += _patch_resize_and_pad_transforms(
                pipeline, target_h, target_w)
        if patched_total == 0:
            logger.warning(
                'No Resize/Pad transform found to patch for --img-size. '
                'The output ori_shape/pad_shape may remain unchanged.')
        else:
            logger.info('Patched %d Resize/Pad transform entries for --img-size=(%d,%d).',
                        patched_total, target_h, target_w)

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    # FLOPs / speed: always skip HF pretrain at build; load cfg.load_from ckpt if set.
    strip_pretrained_for_flops(cfg, logger=logger)
    load_from = _resolve_load_from(cfg, args)
    if args.speed_test:
        _enable_last_layer_head_only_in_cfg(cfg, logger=logger)
    result = {}
    result['load_from'] = load_from or 'N/A'
    avg_flops = []
    flops_skipped = bool(args.skip_flops)
    flops_skip_reason = '--skip-flops' if args.skip_flops else None
    analyze_flops = not args.skip_flops
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    if load_from:
        _load_model_checkpoint(model, load_from, logger=logger)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model = apply_inference_deploy_opts(model, args, logger)
    model.eval()
    _forward = model.forward

    params = None
    out_table = ''
    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        for sample in data['data_samples']:
            sample.set_metainfo(dict(mode=args.prompt_mode))
        speed_num_classes = int(getattr(args, 'speed_num_classes', 0) or 0)
        if speed_num_classes > 0:
            _limit_speed_test_num_classes(
                data['data_samples'],
                speed_num_classes,
                device=_tensor_device_from_batch_inputs(data['inputs']))
        model.forward = partial(_forward, data_samples=data['data_samples'])
        if analyze_flops:
            try:
                outputs = get_model_complexity_info(
                    model,
                    None,
                    inputs=data['inputs'],
                    show_table=args.show_table,
                    show_arch=False,
                )
                avg_flops.append(outputs['flops'])
                params = outputs['params']
                out_table = outputs['out_table']
            except Exception as exc:
                if not args.speed_test:
                    raise
                hint = _flops_failure_hint(exc)
                logger.warning(
                    'get_model_complexity_info failed (%s); disabling FLOPs analysis '
                    'for remaining batches (speed benchmark will still run).%s',
                    exc, hint)
                analyze_flops = False
                flops_skipped = True
                flops_skip_reason = f'{type(exc).__name__}: {exc}{hint}'
                with torch.no_grad():
                    model(data['inputs'])
                params = parameter_count(model)['']
                out_table = (
                    f'\n(FLOPs / activation table skipped: {flops_skip_reason})\n')
        else:
            with torch.no_grad():
                model(data['inputs'])
            if params is None:
                params = parameter_count(model)['']
            if not out_table.strip():
                reason = flops_skip_reason or 'not requested'
                out_table = f'\n(FLOPs / activation table skipped: {reason})\n'
        result['compute_type'] = 'dataloader: load a picture from the dataset'
    del data_loader

    if flops_skipped or not avg_flops:
        mean_flops = 'N/A'
        if params is None:
            params = parameter_count(model)['']
    else:
        mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params
    result['out_table'] = out_table

    if args.speed_test:
        speed_result = benchmark_speed(args, model, _forward, cfg, logger)
        result.update(speed_result)

    try:
        result['module_params_table'] = _build_depth1_params_table(model)
    except Exception as exc:
        logger.warning('depth-1 params table failed: %s', exc)
        result['module_params_table'] = ''

    return result


def _synchronize_if_needed(use_cuda: bool):
    if use_cuda:
        torch.cuda.synchronize()


def _tensor_device_from_batch_inputs(inputs) -> torch.device:
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
        return inputs[0].device
    return torch.device('cpu')


def _resolve_speed_device(sample,
                          fallback: Optional[torch.device] = None
                          ) -> torch.device:
    gt = getattr(sample, 'gt_instances', None)
    if gt is not None:
        for attr in ('labels', 'bboxes'):
            t = getattr(gt, attr, None)
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                return t.device
    if fallback is not None:
        return fallback
    return torch.device('cpu')


def _limit_speed_test_num_classes(
        data_samples,
        num_classes: int,
        device: Optional[torch.device] = None) -> None:
    """Keep at most ``num_classes``; trim GT only (no fake GT padding)."""
    if num_classes <= 0:
        return
    for sample in data_samples:
        dev = _resolve_speed_device(sample, device)
        gt = sample.gt_instances
        text = sample.get('text', None)
        if gt is not None and len(gt) > 0 and gt.labels is not None and len(
                gt.labels) > 0:
            labels = gt.labels.to(dev)
            keep_old = torch.sort(torch.unique(labels)).values[:num_classes]
            keep_t = torch.tensor(
                [int(x) for x in keep_old.tolist()], device=dev, dtype=labels.dtype)
            mask = torch.isin(labels, keep_t)
            labels = labels[mask]
            bboxes = None
            if getattr(gt, 'bboxes', None) is not None and len(gt.bboxes) == len(
                    gt.labels):
                bboxes = gt.bboxes.to(dev, dtype=torch.float32)[mask]
            new_labels = labels.clone()
            for new_idx, old_lbl in enumerate(keep_old.tolist()):
                new_labels[labels == old_lbl] = new_idx
            new_gt = InstanceData()
            if bboxes is not None:
                new_gt.bboxes = bboxes
            new_gt.labels = new_labels
            sample.gt_instances = new_gt
            if isinstance(text, (list, tuple)):
                keep_ids = [int(x) for x in keep_old.tolist()]
                new_text = [
                    text[i] for i in keep_ids if 0 <= i < len(text)
                ]
                if new_text:
                    sample.set_metainfo(dict(text=new_text))
        elif isinstance(text, (list, tuple)) and len(text) > num_classes:
            sample.set_metainfo(dict(text=list(text[:num_classes])))


def _count_speed_sample_num_classes(sample) -> int:
    """Distinct class count per image (after optional ``--speed-num-classes`` cap).

    Prefer ``gt_instances.labels`` unique count (actual categories on the image).
    Fall back to text / tokens only when GT is unavailable.
    """
    gt = getattr(sample, 'gt_instances', None)
    if (gt is not None and len(gt) > 0 and gt.labels is not None
            and len(gt.labels) > 0):
        return int(torch.unique(gt.labels).numel())
    text = sample.get('text', None)
    if isinstance(text, (list, tuple)):
        return len(text)
    if isinstance(text, str) and text.strip():
        tokens_positive = sample.get('tokens_positive', None)
        if isinstance(tokens_positive, dict) and tokens_positive:
            return len(tokens_positive)
        parts = [
            p for p in text.strip('. ').split('. ')
            if p.strip()
        ]
        return max(len(parts), 1)
    return 0


def _mean_classes_per_image(data_samples) -> float:
    if not data_samples:
        return 0.0
    counts = [_count_speed_sample_num_classes(s) for s in data_samples]
    return float(sum(counts)) / len(counts)


def benchmark_speed(args, model, _forward, cfg, logger):
    use_cuda = torch.cuda.is_available()
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    data_iter = iter(data_loader)
    # Outer perf_counter around _forward (includes extra cuda.sync vs model wall).
    latencies_ms = []
    # OPUS forward_wall_ms inside _forward (model-only wall clock, default for FPS).
    model_wall_ms = []
    module_latency_samples = {}
    # Per-iter sum of segment times (same keys as OPUS gap_unaccounted).
    # mean of these sums avoids sum(mean(seg_k)) != mean(sum(seg_k)) when keys
    # are missing in some iterations (e.g. vpg vs vpg_prepare branch).
    segment_sum_per_iter = []
    classes_per_image = []

    total_iters = args.warmup_iter + args.speed_iter
    if total_iters <= 0:
        raise ValueError('warmup_iter + speed_iter must be > 0')

    speed_num_classes = int(getattr(args, 'speed_num_classes', 0) or 0)
    if speed_num_classes > 0:
        logger.info(
            'speed benchmark: keep at most %d class(es) per sample '
            '(--speed-num-classes; gt + text filtered)',
            speed_num_classes)

    for i in range(total_iters):
        try:
            data_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data_batch = next(data_iter)

        data = model.data_preprocessor(data_batch)
        for sample in data['data_samples']:
            sample.set_metainfo(dict(mode=args.prompt_mode))
        if speed_num_classes > 0:
            _limit_speed_test_num_classes(
                data['data_samples'],
                speed_num_classes,
                device=_tensor_device_from_batch_inputs(data['inputs']))
        model.forward = partial(_forward, data_samples=data['data_samples'])

        _synchronize_if_needed(use_cuda)
        start_t = time.perf_counter()
        with torch.no_grad():
            # BaseDetector.forward does not accept speed_test; some detectors
            # (e.g. OPUS) implement it on _forward only.
            _fwd = getattr(model, '_forward', None)
            _sig = inspect.signature(_fwd) if _fwd is not None else None
            use_trex_speed = (
                _fwd is not None and _sig is not None
                and 'speed_test' in _sig.parameters
                and args.measure_mode == 'tensor')
            if use_trex_speed:
                call_kw = dict(speed_test=True)
                if 'speed_profile_segments' in _sig.parameters:
                    call_kw['speed_profile_segments'] = (
                        not getattr(args, 'no_segment_timing', False))
                _fwd(data['inputs'], data['data_samples'], **call_kw)
            else:
                model(data['inputs'], mode=args.measure_mode)
        _synchronize_if_needed(use_cuda)
        end_t = time.perf_counter()

        if i >= args.warmup_iter:
            latencies_ms.append((end_t - start_t) * 1000.0)
            classes_per_image.append(
                _mean_classes_per_image(data['data_samples']))
            stats = getattr(model, 'latest_speed_stats', None)
            if isinstance(stats, dict) and 'forward_wall_ms' in stats:
                model_wall_ms.append(float(stats['forward_wall_ms']))
            if isinstance(stats, dict):
                if not getattr(args, 'no_segment_timing', False):
                    if any(k not in SKIP_SUM_KEYS for k in stats):
                        segment_sum_per_iter.append(
                            sum(float(v) for k, v in stats.items()
                                if k not in SKIP_SUM_KEYS))
                for k, v in stats.items():
                    module_latency_samples.setdefault(k, []).append(float(v))

    del data_loader

    if len(latencies_ms) == 0:
        raise RuntimeError('No latency samples collected. Increase speed-iter.')

    latencies_ms = np.array(latencies_ms, dtype=np.float64)
    wrapper_mean = float(np.mean(latencies_ms))
    wall_arr = np.array(model_wall_ms, dtype=np.float64)
    use_model_wall = wall_arr.size == latencies_ms.size and wall_arr.size > 0
    if use_model_wall:
        mean_ms = float(np.mean(wall_arr))
        p50_ms = float(np.percentile(wall_arr, 50))
        p90_ms = float(np.percentile(wall_arr, 90))
        p95_ms = float(np.percentile(wall_arr, 95))
        latency_source = 'model_forward_wall'
    else:
        mean_ms = wrapper_mean
        p50_ms = float(np.percentile(latencies_ms, 50))
        p90_ms = float(np.percentile(latencies_ms, 90))
        p95_ms = float(np.percentile(latencies_ms, 95))
        latency_source = 'benchmark_wrapper'
    fps_e2e = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    avg_classes = float(np.mean(classes_per_image)) if classes_per_image else 0.0

    logger.info(
        'speed: fps_e2e=%.2f e2e_mean=%.3fms p50=%.3f p95=%.3f '
        'avg_classes_per_image=%.2f src=%s%s',
        fps_e2e, mean_ms, p50_ms, p95_ms, avg_classes, latency_source,
        f' outer={wrapper_mean:.3f}ms' if use_model_wall else '')
    speed_result = dict(
        speed_mode=args.measure_mode,
        speed_prompt_mode=args.prompt_mode,
        speed_num_classes=speed_num_classes,
        avg_classes_per_image=avg_classes,
        speed_warmup_iter=args.warmup_iter,
        speed_iter=args.speed_iter,
        latency_e2e_mean_ms=mean_ms,
        latency_mean_ms=mean_ms,
        latency_p50_ms=p50_ms,
        latency_p90_ms=p90_ms,
        latency_p95_ms=p95_ms,
        fps_e2e=fps_e2e,
        fps=fps_e2e,
        latency_source=latency_source)
    if use_model_wall:
        speed_result['wrapper_latency_mean_ms'] = wrapper_mean
    if len(module_latency_samples) > 0:
        module_stats = {}
        for name, values in module_latency_samples.items():
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0:
                continue
            module_stats[name] = dict(
                mean_ms=float(np.mean(arr)),
                p50_ms=float(np.percentile(arr, 50)),
                p90_ms=float(np.percentile(arr, 90)),
                p95_ms=float(np.percentile(arr, 95)))
        speed_result['module_speed_stats'] = module_stats
        if len(segment_sum_per_iter) > 0:
            seg_arr = np.array(segment_sum_per_iter, dtype=np.float64)
            speed_result['segment_sum_mean_ms'] = float(np.mean(seg_arr))
            speed_result['segment_sum_p50_ms'] = float(np.percentile(seg_arr, 50))
        if module_stats.get('cache_prompt_ms'):
            m = module_stats['cache_prompt_ms']['mean_ms']
            speed_result['latency_cache_prompt_mean_ms'] = m
            speed_result['fps_cache_prompt'] = 1000.0 / m if m > 0 else 0.0
        if module_stats.get('interactive_ms'):
            m = module_stats['interactive_ms']['mean_ms']
            speed_result['latency_interactive_mean_ms'] = m
            speed_result['fps_interactive'] = 1000.0 / m if m > 0 else 0.0
        if (speed_result.get('fps_cache_prompt') is not None
                or speed_result.get('fps_interactive') is not None):
            _fc = speed_result.get('fps_cache_prompt')
            _fi = speed_result.get('fps_interactive')
            logger.info(
                'speed (derived): fps_cache_prompt=%s fps_interactive=%s',
                f'{_fc:.2f}' if _fc is not None else 'n/a',
                f'{_fi:.2f}' if _fi is not None else 'n/a')
    return speed_result


def _format_speed_analysis(result: dict, verbose: bool) -> str:
    """Long-form interpretation (only when verbose=True)."""
    if not verbose or 'latency_e2e_mean_ms' not in result:
        return ''
    mean_fwd = result.get('latency_e2e_mean_ms', result.get('latency_mean_ms'))
    p50 = result['latency_p50_ms']
    p95 = result['latency_p95_ms']
    src = result.get('latency_source', 'benchmark_wrapper')
    lines = ['--- Interpretation (verbose) ---']
    if src == 'model_forward_wall':
        lines.extend([
            f'Model forward wall (OPUS _forward; used for FPS only): '
            f'mean={mean_fwd:.3f} ms, p50={p50:.3f} ms, p95={p95:.3f} ms.',
            f'p95/mean={p95 / mean_fwd:.3f}: values >>1 mean noticeable tail latency '
            f'(scheduler, sync, occasional slower kernels).',
            'Timed region excludes dataloader and data_preprocessor.',
        ])
        wmean = result.get('wrapper_latency_mean_ms')
        if wmean is not None:
            lines.append(
                f'Outer benchmark timer (NOT used for FPS): mean={wmean:.3f} ms; '
                f'overhead vs model wall ~{wmean - mean_fwd:.3f} ms.')
    else:
        lines.extend([
            f'Outer timer (cuda.sync + forward + sync; no OPUS forward_wall_ms): '
            f'mean={mean_fwd:.3f} ms, p50={p50:.3f} ms, p95={p95:.3f} ms.',
            f'p95/mean={p95 / mean_fwd:.3f}: values >>1 mean noticeable tail latency '
            f'(scheduler, sync, occasional slower kernels).',
            'Timed region excludes dataloader and data_preprocessor.',
        ])
    mods = result.get('module_speed_stats')
    if mods:
        ranked = sorted(
            ((k, v['mean_ms']) for k, v in mods.items()
             if k not in SKIP_SUM_KEYS),
            key=lambda x: -x[1])
        total_labeled = sum(
            v['mean_ms'] for k, v in mods.items() if k not in SKIP_SUM_KEYS)
        if ranked:
            top_name, top_ms = ranked[0]
            lines.append(
                f'Largest module by mean latency: {top_name} ({top_ms:.3f} ms).')
        seg_sum_mean = result.get('segment_sum_mean_ms')
        if seg_sum_mean is not None:
            lines.append(
                f'Mean per-iter sum of timed segments: {seg_sum_mean:.3f} ms '
                f'(matches gap_unaccounted identity; prefer over sum of per-module means).')
        elif total_labeled > 0:
            lines.append(
                f'Sum of per-module means: {total_labeled:.3f} ms '
                f'(biased if segment keys differ across iterations).')
        if 'forward_wall_ms' in mods and src != 'model_forward_wall':
            fwm = mods['forward_wall_ms']['mean_ms']
            delta_outer = mean_fwd - fwm
            lines.append(
                f'Forward wall inside _forward (mean): {fwm:.3f} ms; '
                f'outer timer minus forward wall: {delta_outer:.3f} ms '
                f'(first cuda.sync + early Python before t_wall_start).')
            if seg_sum_mean is not None and 'gap_unaccounted_ms' in mods:
                gi = mods['gap_unaccounted_ms']['mean_ms']
                reconc_gap = fwm - seg_sum_mean
                lines.append(
                    f'Forward wall minus mean per-iter segment sum: {reconc_gap:.3f} ms '
                    f'(should match mean gap_unaccounted_ms={gi:.3f} ms).')
        if 'gap_unaccounted_ms' in mods:
            gi = mods['gap_unaccounted_ms']['mean_ms']
            lines.append(
                f'gap_unaccounted_ms (mean): {gi:.3f} ms — residual inside forward_wall '
                f'after summing timed segments each iteration.')
        if seg_sum_mean is not None and abs(total_labeled - seg_sum_mean) > 0.05:
            lines.append(
                f'Note: sum of per-module means ({total_labeled:.3f} ms) vs mean per-iter sum '
                f'({seg_sum_mean:.3f} ms) differ by '
                f'{abs(total_labeled - seg_sum_mean):.3f} ms (sparse keys or branch variance).')
    lines.append('')
    return '\n'.join(lines)


_MODULE_STAT_ORDER = [
    'e2e_ms',
    'cache_prompt_ms',
    'interactive_ms',
    'backbone', 'neck', 'prompt_prep',
    'misc_text_len_check', 'text_encoder',
    'pre_transformer',
    'vpg_prepare', 'vpg_roi_align', 'visual_prompt_generator',
    'vpg',
    'encoder', 'pre_decoder', 'decoder',
    'misc_merge_head_inputs', 'head',
    'gap_unaccounted_ms',
]


def build_report_text(args, result: dict) -> str:
    """Compact tables by default; --show-table includes full FlopAnalyzer ASCII2 table."""
    split_line = '=' * 42
    verbose = getattr(args, 'verbose_report', False)
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    out_table = str(result['out_table'])

    parts = [split_line, 'FLOPs / params summary', split_line]
    if pad_shape != ori_shape:
        parts.append(
            f'Padded {_shape_hw_str(pad_shape)} vs original {_shape_hw_str(ori_shape)} '
            f'[order is H,W like tensor [..., H, W], not W,H]')
    parts.append(
        _fmt_table(
            ['Compute', 'Input shape', 'FLOPs', 'Params'],
            [[compute_type, _shape_hw_str(pad_shape), flops, params]]))
    lf = result.get('load_from', 'N/A')
    if lf and lf != 'N/A':
        parts.append(f'Checkpoint (load_from): {lf}')

    mp = result.get('module_params_table', '').strip()
    if mp:
        parts.extend([
            '',
            split_line,
            'Module parameters (depth-1 child subtrees, non-overlapping)',
            split_line,
            mp,
        ])

    _omit_detail = (
        not verbose and not args.show_table
        and len(out_table.strip()) > 400)
    if _omit_detail:
        parts.append(
            '(FlopAnalyzer per-layer table omitted; use --show-table or '
            '--verbose-report.)')
    else:
        if out_table.strip():
            parts.extend([
                '',
                split_line,
                'Per-module FLOPs / activations (FlopAnalyzer)',
                split_line,
                out_table.rstrip(),
            ])

    if args.speed_test:
        parts.extend(['', split_line, 'Speed', split_line])
        _src = result.get('latency_source', 'unknown')
        if _src == 'model_forward_wall':
            _src_short = 'model_forward_wall'
        else:
            _src_short = 'benchmark_wrapper'

        speed_rows = [
            ['measure_mode', result['speed_mode']],
            ['prompt_mode', result['speed_prompt_mode']],
            ['speed_num_classes', result.get('speed_num_classes', 0)],
            ['avg_classes_per_image',
             f'{result.get("avg_classes_per_image", 0):.2f}'],
        ]
        speed_rows.extend([
            ['warmup_iters', result['speed_warmup_iter']],
            ['measured_iters', result['speed_iter']],
            ['latency_source', _src_short],
            ['fps_e2e', f'{result.get("fps_e2e", result.get("fps", 0)):.2f}'],
            ['latency_e2e_mean_ms',
             f'{result.get("latency_e2e_mean_ms", result.get("latency_mean_ms", 0)):.3f}'],
            ['latency_p50_ms', f'{result["latency_p50_ms"]:.3f}'],
        ])
        if getattr(args, 'decoder_early_exit_layer', None) is not None:
            speed_rows.append([
                'decoder_early_exit_layer',
                int(args.decoder_early_exit_layer),
            ])
        if verbose:
            speed_rows.append(
                ['latency_p90_ms', f'{result["latency_p90_ms"]:.3f}'])
        speed_rows.append(['latency_p95_ms', f'{result["latency_p95_ms"]:.3f}'])
        if result.get('wrapper_latency_mean_ms') is not None:
            speed_rows.append([
                'outer_timer_ms (not FPS)',
                f'{result["wrapper_latency_mean_ms"]:.3f}',
            ])
        if result.get('segment_sum_mean_ms') is not None:
            speed_rows.append([
                'segment_sum_mean_ms',
                f'{result["segment_sum_mean_ms"]:.3f}',
            ])
            speed_rows.append([
                'segment_sum_p50_ms',
                f'{result.get("segment_sum_p50_ms", 0):.3f}',
            ])
        if result.get('fps_cache_prompt') is not None:
            speed_rows.append([
                'fps_cache_prompt',
                f'{result["fps_cache_prompt"]:.2f}',
            ])
            if result.get('latency_cache_prompt_mean_ms') is not None:
                speed_rows.append([
                    'latency_cache_prompt_mean_ms',
                    f'{result["latency_cache_prompt_mean_ms"]:.3f}',
                ])
        if result.get('fps_interactive') is not None:
            speed_rows.append([
                'fps_interactive',
                f'{result["fps_interactive"]:.2f}',
            ])
            if result.get('latency_interactive_mean_ms') is not None:
                speed_rows.append([
                    'latency_interactive_mean_ms',
                    f'{result["latency_interactive_mean_ms"]:.3f}',
                ])

        parts.append(_fmt_table(['Setting', 'Value'], speed_rows))
        if not verbose:
            parts.append(
                '(fps_e2e: full _forward wall; fps_cache_prompt: segments excluding '
                'prompt encoding; fps_interactive: vpg+pre_decoder+decoder+head. '
                'avg_classes_per_image: mean distinct GT class count per image '
                '(unique gt_instances.labels; fallback to text entity count when '
                'no GT) over measured iters, after --speed-num-classes cap. '
                'Timed region excludes dataloader + data_preprocessor.)')
        else:
            parts.append(
                'fps_* = 1000 / corresponding latency_*_mean_ms when segment timing '
                'is enabled.')

        analysis = _format_speed_analysis(result, verbose)
        if analysis:
            parts.extend(['', analysis.rstrip()])

        mods = result.get('module_speed_stats')
        if mods:
            if verbose:
                mod_headers = ['module', 'mean', 'p50', 'p90', 'p95']
                mod_rows = []
                for name in _MODULE_STAT_ORDER:
                    if name not in mods:
                        continue
                    stat = mods[name]
                    mod_rows.append([
                        name,
                        f'{stat["mean_ms"]:.3f}',
                        f'{stat["p50_ms"]:.3f}',
                        f'{stat["p90_ms"]:.3f}',
                        f'{stat["p95_ms"]:.3f}',
                    ])
            else:
                mod_headers = ['module', 'mean', 'p50', 'p95']
                mod_rows = []
                for name in _MODULE_STAT_ORDER:
                    if name not in mods:
                        continue
                    stat = mods[name]
                    mod_rows.append([
                        name,
                        f'{stat["mean_ms"]:.3f}',
                        f'{stat["p50_ms"]:.3f}',
                        f'{stat["p95_ms"]:.3f}',
                    ])
            if mod_rows:
                parts.extend(['', 'Segment / module latency (ms)'])
                parts.append(_fmt_table(mod_headers, mod_rows))

    parts.extend([
        '',
        split_line,
        'Disclaimer: verify FLOPs coverage and op support before citing in papers.',
        split_line,
    ])
    return '\n'.join(parts)


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    resolve_output_paths(args, logger)
    logger.info(
        'Outputs: out_dir=%s report=%s',
        args.out_dir, args.report_out_file)
    result = inference(args, logger)
    split_line = '=' * 30

    file_header = ''
    if args.report_out_file:
        file_header = (
            f'Generated: {datetime.datetime.now().isoformat(timespec="seconds")}\n'
            f'Config: {args.config}\n{split_line}\n\n')

    report_body = build_report_text(args, result)
    full_for_file = file_header + report_body if args.report_out_file else report_body

    print(report_body)
    if args.report_out_file:
        out_path = Path(args.report_out_file).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_for_file, encoding='utf-8')
        logger.info('Report written to %s', out_path)


if __name__ == '__main__':
    main()
