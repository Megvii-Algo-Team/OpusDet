# Copyright (c) OpenMMLab. All rights reserved.
"""Aggregate per-dataset metrics from flat evaluation JSON.

Supports:

- **ODinW / COCO (multi-dataset)**: ``{dataset}/coco/bbox_mAP``, …
- **COCO (single eval, e.g. O365)**: flat ``coco/bbox_mAP``, …
- **LVIS / mini-LVIS**: ``lvis_fixed_ap/AP``, ``lvis_fixed_ap/AP50``, … (any
  ``lvis*/…`` block; full and mini share the same shape)
- **RefExp / RefCOCO**: keys ``{prefix}/refexp/{metric}``, …

RefExp notes:

- **Prefix** (before ``/refexp/``) names the **eval split / group**, not the
  metric family by itself. Examples: ``refcoco_testA/refexp/...`` is **RefCOCO**
  benchmark’s **testA**; ``refcoco+_val`` is RefCOCO+’s val; ``refcocog_test``
  is RefCOCOg’s test. The special prefix **``val``** is a **single concat row**
  holding RefCOCO + RefCOCO+ + RefCOCOg **val** numbers together (same layout
  as merging the three val columns). It is **not** the same as averaging
  per-benchmark split rows.
- **Metric** tail ``refcoco_precision@1``, ``refcocog_precision@1``, … names
  **which benchmark** the score belongs to (RefCOCO / RefCOCO+ / RefCOCOg) and
  **precision@k**.
- **``mean_precision``** and **0.0** placeholders are **omitted** from tables
  and averages (0 = not evaluated for that benchmark on that split).
- **Summary** and **precision@1** average only over mandated splits: RefCOCO &
  RefCOCO+ use ``refcoco_val`` / ``refcoco_testA`` / ``refcoco_testB`` (and
  ``refcoco+_*``); RefCOCOg uses ``refcocog_val`` / ``refcocog_test``. If a
  per-benchmark **val** row (e.g. ``refcoco_val``) is **absent**, that slot is
  filled from the concat row ``val/refexp/...`` (same metric key:
  ``refcoco_precision@*``, ``refcoco+_precision@*``, ``refcocog_precision@*``).
  The concat row is not averaged as its own split. ``--include-val-aggregate``
  still controls fallback for extra keys only.
- **Per-split** table: columns are ``RefCOCO precision@1`` … only (no
  ``mean_precision``).

Example:

  # ODinW-35 mean (macro avg over 35 subsets)
  python tools/json_metrics_avg.py \\
    work_dirs/test/opus_dinov3_convnext-b/odinw35/text_only/<ts>/<ts>.json \\
    --ignore-invalid

  python tools/json_metrics_avg.py metrics.json --format markdown -o report
  # writes report.md

  # Auto mode picks coco / refexp / lvis from key names (ODinW ``{ds}/coco/*``,
  # O365 ``coco/*``, LVIS ``lvis_fixed_ap/*``).

  Batch under OPUS eval tree (keep latest timestamp per bmk/mode)::

    python tools/json_metrics_avg.py --ignore-invalid --format markdown \\
      --scan-root work_dirs/test --name-substr 202608 --out-dir work_dirs/test/summary \\
      --latest-per-benchmark

  Batch output names default to ``{bmk}_{mode}[__{H}x{W}][__decxN].md`` (``bmk`` from work dir,
  ``mode`` from sibling ``.log`` e.g. ``mode='visual.I.1'``). Resolution ``{H}x{W}`` is parsed
  from ``test_dataloader`` in the same ``.log`` (first eval ``FixScaleResize`` / ``Resize``
  ``scale=``), inserted so different eval scales do not overwrite. If
  ``decoder_early_exit_layer=N`` exists in the log, ``__decxN`` is appended last.
  If the log has no ``mode=`` line, ``mode`` defaults to ``text_only``. Use
  ``--legacy-batch-names`` for the previous long path-based filenames.

  Reports also print ``Checkpoint (load_from): ...`` from the ``load_from = '...'``
  line in the same ``.log`` when present.
"""
import argparse
import json
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

_METRICS_INDEX_LOCK = threading.Lock()


def _metrics_index_cache_path(metrics_root: Path) -> Path:
    """Disk snapshot beside ``metrics/summary/_storage_cache``."""
    root = metrics_root.resolve()
    return root / "summary" / "_storage_cache" / "metrics_index.json"


def _deserialize_metrics_index(payload: Dict[str, Any]) -> Tuple[
    float, List[Path], Dict[str, List[Path]], List[Path]
]:
    version = float(payload.get("version", 0.0))
    catalog = [Path(p) for p in payload.get("catalog", [])]
    by_train_dir = {
        str(k): [Path(p) for p in paths]
        for k, paths in payload.get("by_train_dir", {}).items()
    }
    no_train_dir = [Path(p) for p in payload.get("no_train_dir", [])]
    return version, catalog, by_train_dir, no_train_dir


def _serialize_metrics_index(
    built: Tuple[float, List[Path], Dict[str, List[Path]], List[Path]],
) -> Dict[str, Any]:
    version, catalog, by_train_dir, no_train_dir = built
    return {
        "version": version,
        "catalog_count": len(catalog),
        "catalog": [str(p) for p in catalog],
        "by_train_dir": {
            k: [str(p) for p in paths] for k, paths in by_train_dir.items()
        },
        "no_train_dir": [str(p) for p in no_train_dir],
    }


def clear_metrics_index_cache(metrics_root: Optional[Path] = None) -> None:
    """Remove on-disk metrics JSON catalog snapshot."""
    if metrics_root is None:
        return
    cache_path = _metrics_index_cache_path(metrics_root)
    try:
        cache_path.unlink(missing_ok=True)
    except OSError:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description='Parse evaluation JSON: COCO (*/coco/*) or RefExp (*/refexp/*).')
    p.add_argument(
        'json_path',
        type=str,
        nargs='?',
        default=None,
        help='path to metrics .json (omit when using --scan-root)',
    )
    p.add_argument(
        '--mode',
        choices=('auto', 'coco', 'refexp', 'lvis'),
        default='auto',
        help='metric key layout (default: detect from JSON)')
    p.add_argument(
        '--ignore-invalid',
        action='store_true',
        help='[coco] skip -1 when averaging (COCO N/A)')
    p.add_argument(
        '--ignore-zero',
        action='store_true',
        default=None,
        help='[refexp] skip 0 when averaging each metric (default: True for refexp)')
    p.add_argument(
        '--no-ignore-zero',
        action='store_true',
        help='[refexp] include zeros in averages (usually wrong for RefCOCO tables)')
    p.add_argument(
        '--include-val-aggregate',
        action='store_true',
        help='[refexp] include ``val/`` prefix in averages (duplicates concat summary)')
    p.add_argument(
        '--format',
        choices=('plain', 'markdown', 'csv'),
        default='plain',
        help='table style')
    p.add_argument(
        '--no-per-dataset',
        action='store_true',
        help='only print summary table, not full per-split matrix')
    p.add_argument(
        '-o',
        '--out-file',
        type=str,
        default=None,
        metavar='PATH',
        help='write report to this path (UTF-8); extension is chosen from '
        '--format: .txt (plain), .md (markdown), .csv (csv); default is stdout')
    p.add_argument(
        '--scan-root',
        type=str,
        default=None,
        metavar='DIR',
        help='scan root for batch / experiment-name modes '
        '(default for --experiment-name: work_dirs/metrics)',
    )
    p.add_argument(
        '--name-substr',
        type=str,
        default=None,
        metavar='STR',
        help='batch: keep only JSON files whose name contains this substring '
        '(e.g. 0407 or 20260407)',
    )
    p.add_argument(
        '--out-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='batch: write one report per JSON under this directory '
        '(default: {bmk}_{mode}[__HxW][__decxN][__ckpt_<checkpoint>] from .log + folder; '
        '__HxW from test_dataloader eval resize when parseable; '
        '__ckpt_ from load_from basename; mode defaults to text_only; see --legacy-batch-names)',
    )
    p.add_argument(
        '--latest-per-benchmark',
        action='store_true',
        help='batch: for each (benchmark folder, eval mode, decx, resolution, ckpt '
        'from .log) keep only the run with the largest <timestamp> folder; '
        'distinct modes / scales / checkpoints do not overwrite each other',
    )
    p.add_argument(
        '--legacy-batch-names',
        action='store_true',
        help='batch: use long path-based filenames (default: {bmk}_{mode}[__decxN] from '
        'log + benchmark folder)',
    )
    p.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        metavar='EXP',
        help='auto collect metrics under --scan-root (or default work_dirs/metrics) by matching EXP in '
        'sibling .log load_from path',
    )
    p.add_argument(
        '--metrics-root',
        type=str,
        default=None,
        metavar='DIR',
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        '--dump-json',
        action='store_true',
        help='for --experiment-name: write aggregated result as JSON '
        '(default is text summary)',
    )
    return p.parse_args()


FORMAT_SUFFIX = {
    'plain': '.txt',
    'markdown': '.md',
    'csv': '.csv',
}

TIMESTAMP_DIR_RE = re.compile(r'^\d{8}_\d{6}$')


def out_path_for_format(out_file: str, table_format: str) -> Path:
    """Use stem/dir from ``out_file``; suffix always matches ``table_format``."""
    return Path(out_file).with_suffix(FORMAT_SUFFIX[table_format])


def discover_json_files(scan_root: Path, name_substr: str) -> List[Path]:
    """All ``*.json`` under ``scan_root`` whose filename contains ``name_substr``."""
    root = scan_root.resolve()
    out: List[Path] = []
    for p in sorted(root.rglob('*.json')):
        if name_substr in p.as_posix():
            out.append(p)
    return out


def is_raw_metrics_json(path: Path) -> bool:
    """True for raw eval outputs like ``.../<timestamp>/<timestamp>.json``."""
    parent = path.parent.name
    stem = path.stem
    if not TIMESTAMP_DIR_RE.match(parent):
        return False
    if not TIMESTAMP_DIR_RE.match(stem):
        return False
    # exclude vis_data/config-like json in run subfolders
    if 'vis_data' in path.parts:
        return False
    return True


def latest_json_per_benchmark(paths: List[Path]) -> List[Path]:
    """One JSON per ``(benchmark_dir, eval_mode, decx, resolution, ckpt)`` with max ``timestamp``.

    ``eval_mode``, ``decoder_early_exit_layer``, eval resolution (from
    ``test_dataloader`` in the sibling ``.log``), and ``load_from`` checkpoint
    (``ckpt_slot_key``) follow ``batch_report_path`` rules, so distinct scales,
    exits, or checkpoints do not collapse to a single ``latest`` slot.
    """
    best: Dict[Tuple[Path, str, str, str, str], Path] = {}
    for p in paths:
        if len(p.parts) < 3:
            continue
        bench = p.parent.parent.resolve()
        log_p = p.with_suffix('.log')
        mode = parse_mode_from_log(log_p)
        if mode is None:
            mode = 'text_only'
        mode_key = mode.strip()
        decx = parse_decoder_early_exit_layer_from_log(log_p)
        decx_key = f"decx{decx}" if decx is not None else "decxNA"
        res = parse_eval_resolution_from_log(log_p)
        res_key = res if res else "resNA"
        ck_key = ckpt_slot_key(parse_load_from_log(log_p))
        key = (bench, mode_key, decx_key, res_key, ck_key)
        cur = best.get(key)
        if cur is None or p.parent.name > cur.parent.name:
            best[key] = p
    return sorted(best.values())


# Model eval ``mode`` in dumped config (same directory as metrics JSON).
MODE_FROM_LOG_RE = re.compile(r"(?m)^\s*mode\s*=\s*['\"]([^'\"]+)['\"]")


def parse_mode_from_log(log_path: Path) -> Optional[str]:
    """Read ``mode='...'`` from ``torchrun`` / mmengine config dump (e.g. line ``mode='visual.I.1'``)."""
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    matches = MODE_FROM_LOG_RE.findall(text)
    if not matches:
        return None
    # First ``mode='...`` at indent in the merged ``model = dict`` dump (runtime
    # mode after ``--cfg-options``); later matches may be submodules.
    return matches[0].strip()


def sanitize_mode_for_filename(mode: str) -> str:
    """Keep ``visual.I.1`` as in the log; strip only unsafe path characters."""
    s = mode.strip()
    return re.sub(r'[/\\:*?"<>|\s]', '_', s)


def ckpt_slot_key(load_from: Optional[str]) -> str:
    """Token from eval ``load_from`` (checkpoint path) for dedup keys and report stems.

    Same benchmark/mode/decx/res with different checkpoints become distinct slots.
    Missing ``load_from`` maps to ``"-"`` (legacy behaviour in one bucket).
    """
    if not load_from or not str(load_from).strip():
        return "-"
    base = Path(str(load_from).strip()).name
    if not base:
        return "-"
    safe = re.sub(r'[/\\:*?"<>|\s]', "_", base)
    safe = re.sub(r"__+", "_", safe).strip("_")
    if not safe:
        return "-"
    return safe[:88]


def normalize_resolution_token(h: int | str, w: int | str) -> str:
    """Canonical resolution token as short-edge x long-edge."""
    hi = int(h)
    wi = int(w)
    a, b = sorted((hi, wi))
    return f'{a}x{b}'


# Merged config line: ``load_from = 'work_dirs/.../iter_380000.pth'``
LOAD_FROM_LOG_RE = re.compile(
    r"(?m)^\s*load_from\s*=\s*['\"]([^'\"]+)['\"]")
DECODER_EARLY_EXIT_RE = re.compile(
    r"(?m)decoder_early_exit_layer\s*=\s*([0-9]+)")


def parse_load_from_log(log_path: Path) -> Optional[str]:
    """Checkpoint path from the dumped ``load_from = '...'`` line in the eval ``.log``."""
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    m = LOAD_FROM_LOG_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def parse_decoder_early_exit_layer_from_log(log_path: Path) -> Optional[int]:
    """Read ``decoder_early_exit_layer=...`` from eval ``.log`` when present."""
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    m = DECODER_EARLY_EXIT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _test_dataloader_config_chunk(text: str) -> Optional[str]:
    """Slice from ``test_dataloader = dict(`` until the next top-level key (exclude ``test_pipeline``)."""
    m = re.search(r"(?ms)^test_dataloader\s*=\s*dict\(", text)
    if not m:
        return None
    tail = text[m.start():]
    m_end = re.search(
        r"(?ms)^(?:test_evaluator|test_pipeline|train_cfg|train_dataloader|optim_wrapper|param_scheduler)\s*=",
        tail[1:],
    )
    if m_end:
        return tail[: m_end.start() + 1]
    return tail[:35000]


def parse_eval_resolution_from_log(log_path: Path) -> Optional[str]:
    """Eval input ``HxW`` token (``HxW`` as in config ``scale=(H, W)``) from ``test_dataloader``.

    Uses the first ``FixScaleResize`` or ``Resize`` transform in the dumped
    ``test_dataloader`` block with a ``scale=(...)`` tuple.
    """
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    chunk = _test_dataloader_config_chunk(text)
    if not chunk:
        return None
    # Trailing comma inside ``(H, W,)`` is common in dumped configs.
    _pair = r"\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)"
    patterns = (
        rf"scale\s*=\s*{_pair}[\s\S]{{0,800}}?type\s*=\s*'FixScaleResize'",
        rf"type\s*=\s*'FixScaleResize'[\s\S]{{0,800}}?scale\s*=\s*{_pair}",
        rf"scale\s*=\s*{_pair}[\s\S]{{0,800}}?type\s*=\s*'Resize'",
        rf"type\s*=\s*'Resize'[\s\S]{{0,800}}?scale\s*=\s*{_pair}",
    )
    for pat in patterns:
        m = re.search(pat, chunk)
        if m:
            return normalize_resolution_token(m.group(1), m.group(2))
    m = re.search(r"img_scale\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)", chunk)
    if m:
        return normalize_resolution_token(m.group(1), m.group(2))
    return None


def parse_train_dir_from_load_from(load_from: str) -> Optional[str]:
    """Extract training work_dir name from checkpoint path in ``load_from``.

    Example:
      ``work_dirs/exp_train_xxx/iter_380000.pth`` -> ``exp_train_xxx``
    """
    s = (load_from or '').strip()
    if not s:
        return None
    p = Path(s)
    parent = p.parent.name if p.parent else ''
    if parent:
        return parent
    return None


_RUNTIME_SUFFIX_TOKENS = frozenset({'amp', 'fp16', 'bf16'})


def _strip_runtime_suffix_tokens(name: str) -> str:
    toks = [t for t in str(name or '').split('_') if t]
    while toks and toks[-1] in _RUNTIME_SUFFIX_TOKENS:
        toks.pop()
    return '_'.join(toks)


def experiment_dir_names_match(exp_key: str, train_dir: str) -> bool:
    """Exact work_dir name match; ignores trailing runtime flags like ``_amp``."""
    a = (exp_key or '').strip()
    b = (train_dir or '').strip()
    if not a or not b:
        return False
    if a == b:
        return True
    return _strip_runtime_suffix_tokens(a) == _strip_runtime_suffix_tokens(b)


def bmk_from_json_path(json_path: Path) -> str:
    """Short BMK tag aligned with playground ``BMK`` keys.

    Layouts:
    - OPUS: ``work_dirs/test/<product>/<bmk>/<mode>/<ts>/<ts>.json``
    - research-style: ``work_dirs/metrics/<...bmk...>/<ts>/<ts>.json``
    """
    mode_or_exp = json_path.parent.parent
    name = mode_or_exp.name
    # OPUS test_common: .../{bmk}/{mode_tag}/timestamp/timestamp.json
    if re.match(r'^(text_only|visual_|text_visual_)', name):
        name = mode_or_exp.parent.name
    nlow = name.lower()
    if name in ('coco', 'lvis', 'lvis-mv', 'odinw13', 'odinw35'):
        return name
    if 'mini-lvis' in nlow or 'mini_lvis' in nlow or nlow == 'lvis-mv':
        return 'lvis-mv'
    if 'zeroshot_lvis' in nlow or nlow == 'lvis':
        return 'lvis'
    if 'zeroshot_coco' in nlow or 'zeroshot-coco' in nlow:
        return 'coco-zs'
    if 'pretrain_o365' in nlow and 'zeroshot' not in nlow:
        return 'coco'
    if 'odinw13' in nlow:
        return 'odinw13'
    if 'odinw35' in nlow:
        return 'odinw35'
    if 'refexp' in nlow or 'zeroshot_refexp' in nlow:
        return 'refcoco'
    if 'det_anything_bmk' in nlow or 'zeroshot_bmk' in nlow:
        return 'bmk1'
    if 'counting' in nlow:
        return 'counting'
    if 'labubu' in nlow:
        return 'labubu'
    safe = re.sub(r'[^\w\-+.]', '_', name).strip('_')
    return safe[:120] if safe else 'unknown'


def batch_report_path(
    json_path: Path,
    scan_root: Path,
    out_dir: Path,
    table_format: str,
    *,
    legacy_long_path: bool = False,
) -> Path:
    """Default: ``{bmk}_{mode}[__HxW][__decxN][__ckpt_<token>].md`` from sibling ``.log`` + benchmark folder; optional long path stem."""
    suf = FORMAT_SUFFIX.get(table_format, '.md')
    if legacy_long_path:
        root = scan_root.resolve()
        try:
            rel = json_path.resolve().relative_to(root)
        except ValueError:
            rel = Path(json_path.name)
        stem = rel.with_suffix('').as_posix().replace('/', '_')
        return out_dir / f'{stem}{suf}'

    log_path = json_path.with_suffix('.log')
    mode = parse_mode_from_log(log_path)
    decx = parse_decoder_early_exit_layer_from_log(log_path)
    res = parse_eval_resolution_from_log(log_path)
    bmk = bmk_from_json_path(json_path)
    if mode is None:
        mode = 'text_only'
    mode_safe = sanitize_mode_for_filename(mode)
    res_part = f'__{res}' if res else ''
    decx_suffix = f'__decx{decx}' if decx is not None else ''
    ck = ckpt_slot_key(parse_load_from_log(log_path))
    ck_suffix = f'__ckpt_{ck}'
    return out_dir / f'{bmk}_{mode_safe}{res_part}{decx_suffix}{ck_suffix}{suf}'


COCO_KEY = re.compile(r'^(.+)/coco/(.+)$')
# Single-eval COCO JSON (e.g. Objects365): ``coco/bbox_mAP`` → normalized below.
COCO_FLAT_KEY = re.compile(r'^coco/(.+)$')
REFEXP_KEY = re.compile(r'^(.+)/refexp/(.+)$')

# Benchmark id in metric tail: ``{prefix}_precision@{k}`` (see module doc).
REFEXP_BENCHMARKS = (
    ('RefCOCO', 'refcoco'),
    ('RefCOCO+', 'refcoco+'),
    ('RefCOCOg', 'refcocog'),
)

# Summary / P@1: mean only over these ``{bench}_{split}`` prefixes.
# Missing ``*_val`` rows borrow metrics from concat ``val/`` (same ``mkey``).
REFEXP_STATS_SPLITS: Dict[str, Tuple[str, ...]] = {
    'refcoco': ('refcoco_val', 'refcoco_testA', 'refcoco_testB'),
    'refcoco+': ('refcoco+_val', 'refcoco+_testA', 'refcoco+_testB'),
    'refcocog': ('refcocog_val', 'refcocog_test'),
}


def refexp_value_for_split(
    by_ds: Dict[str, Dict[str, float]],
    ds: str,
    mkey: str,
) -> Optional[float]:
    """Value for ``ds`` + ``mkey``; if ``*_val`` split row missing, use ``val`` concat row."""
    if ds in by_ds and mkey in by_ds[ds]:
        return float(by_ds[ds][mkey])
    if ds != 'val' and ds.endswith('_val') and 'val' in by_ds and mkey in by_ds['val']:
        return float(by_ds['val'][mkey])
    return None


def refexp_bench_from_metric(metric: str) -> Optional[str]:
    """Map ``refcoco_precision@1`` → ``refcoco``; order matters for ``refcoco+``."""
    if metric.startswith('refcoco+_'):
        return 'refcoco+'
    if metric.startswith('refcocog_'):
        return 'refcocog'
    if metric.startswith('refcoco_'):
        return 'refcoco'
    return None


def refexp_mean_over_mandated_splits(
    by_ds: Dict[str, Dict[str, float]],
    mkey: str,
    splits: Tuple[str, ...],
    ignore_zero: bool,
) -> Tuple[Optional[float], int, int]:
    """Mean over listed split prefixes; returns (mean, n_used, n_skipped_zero)."""
    present: List[float] = []
    for ds in splits:
        v = refexp_value_for_split(by_ds, ds, mkey)
        if v is None:
            continue
        present.append(v)
    raw_n = len(present)
    if ignore_zero:
        used = [x for x in present if x != 0.0]
        m = sum(used) / len(used) if used else None
        return m, len(used), raw_n - len(used)
    m = sum(present) / len(present) if present else None
    return m, raw_n, 0

# Column order for RefExp tables (avoid lexicographic refcoco+ before refcoco).
# ``mean_precision`` is parsed but dropped for RefExp reporting.
REFEXP_METRIC_ORDER = (
    'refcoco_precision@1', 'refcoco_precision@5', 'refcoco_precision@10',
    'refcoco+_precision@1', 'refcoco+_precision@5', 'refcoco+_precision@10',
    'refcocog_precision@1', 'refcocog_precision@5', 'refcocog_precision@10',
)


def order_refexp_metrics(names: List[str]) -> List[str]:
    """Stable order: known metrics first, then any extra keys sorted."""
    known = [m for m in REFEXP_METRIC_ORDER if m in names]
    extra = sorted(
        m for m in names
        if m not in REFEXP_METRIC_ORDER and m != 'mean_precision')
    return known + extra


def refexp_metrics_for_tables(names: List[str]) -> List[str]:
    """RefExp columns: precision@* only; drop ``mean_precision``."""
    return [m for m in order_refexp_metrics(names) if m != 'mean_precision']


def refexp_display_name(internal: str) -> str:
    """Map flat metric key to short display (precision@k, not refcoco_precision@k)."""
    m = re.match(r'^(refcoco\+|refcocog|refcoco)_precision@(\d+)$', internal)
    if m:
        b, r = m.group(1), m.group(2)
        label = {
            'refcoco': 'RefCOCO',
            'refcoco+': 'RefCOCO+',
            'refcocog': 'RefCOCOg',
        }[b]
        return f'{label} precision@{r}'
    return internal


def _is_lvis_flat_key(k: str) -> bool:
    """``lvis_fixed_ap/AP``-style: first segment starts with ``lvis``."""
    if '/' not in k:
        return False
    head, _ = k.split('/', 1)
    return head.startswith('lvis')


# LVIS fixed-AP column order (full / mini share names).
LVIS_METRIC_ORDER = (
    'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'APr', 'APc', 'APf',
)


def order_lvis_metrics(names: List[str]) -> List[str]:
    known = [m for m in LVIS_METRIC_ORDER if m in names]
    extra = sorted(m for m in names if m not in LVIS_METRIC_ORDER)
    return known + extra


def _load_json_compat(path: Path) -> Dict[str, Any]:
    """Load metrics JSON with tolerant fallbacks.

    Supports:
    - strict single JSON object
    - corrupted file with extra trailing bytes (take first JSON object)
    - JSONL / repeated JSON-object logs (merge dict objects in order)
    """
    text = path.read_text(encoding='utf-8', errors='replace')

    # Fast path: strict JSON object.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback 0.5: try trimming trailing garbage after the last `}`.
    # Handles tails like "...}}", "...123}", etc.
    stripped = text.strip()
    if stripped.startswith('{') and '}' in stripped:
        last_rbrace = stripped.rfind('}')
        # Try from the last `}` backward to find a valid JSON object.
        for end in range(last_rbrace + 1, 0, -1):
            if stripped[end - 1] != '}':
                continue
            candidate = stripped[:end]
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj

    # Fallback 1: parse first valid JSON object from start.
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text.lstrip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback 2: JSONL / multi-JSON; merge dicts in file order.
    merged: Dict[str, Any] = {}
    parsed_any = False
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            merged.update(obj)
            parsed_any = True
    if parsed_any:
        return merged

    raise ValueError(f'Cannot parse metrics JSON: {path}')


def load_metrics(path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict]:
    """Return (coco_flat, refexp_flat, lvis_flat, meta)."""
    raw = _load_json_compat(path)
    coco_flat: Dict[str, float] = {}
    refexp_flat: Dict[str, float] = {}
    lvis_flat: Dict[str, float] = {}
    meta: Dict = {}
    for k, v in raw.items():
        if not isinstance(v, (int, float)):
            meta[k] = v
            continue
        fv = float(v)
        if COCO_KEY.match(k):
            coco_flat[k] = fv
        elif (m := COCO_FLAT_KEY.match(k)):
            # Reuse ``dataset/coco/metric`` pivot (one pseudo-dataset ``single``).
            coco_flat[f'single/coco/{m.group(1)}'] = fv
        elif REFEXP_KEY.match(k):
            refexp_flat[k] = fv
        elif _is_lvis_flat_key(k):
            lvis_flat[k] = fv
        else:
            meta[k] = fv
    return coco_flat, refexp_flat, lvis_flat, meta


def pivot_flat(flat: Dict[str, float], pattern: re.Pattern):
    """prefix -> metric_name -> value."""
    by_ds: Dict[str, Dict[str, float]] = defaultdict(dict)
    metric_names = set()
    for k, v in flat.items():
        m = pattern.match(k)
        if not m:
            continue
        ds, metric = m.group(1), m.group(2)
        by_ds[ds][metric] = v
        metric_names.add(metric)
    return by_ds, sorted(metric_names)


def pivot_lvis_flat(flat: Dict[str, float]):
    """``lvis_fixed_ap/AP`` -> dataset ``lvis_fixed_ap``, metric ``AP``."""
    by_ds: Dict[str, Dict[str, float]] = defaultdict(dict)
    metric_names = set()
    for k, v in flat.items():
        if '/' not in k:
            continue
        ds, metric = k.split('/', 1)
        if not ds.startswith('lvis'):
            continue
        by_ds[ds][metric] = v
        metric_names.add(metric)
    return by_ds, order_lvis_metrics(sorted(metric_names))


def mean(vals: List[float], ignore_invalid: bool) -> Optional[float]:
    if ignore_invalid:
        vals = [x for x in vals if x != -1.0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt_cell(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return 'n/a'
    return f'{x:.{nd}f}'


def print_table(
    headers: List[str],
    rows: List[List[str]],
    table_format: str,
    out=sys.stdout,
):
    if table_format == 'csv':
        out.write(','.join(headers) + '\n')
        for row in rows:
            out.write(','.join(row) + '\n')
        return
    if table_format == 'markdown':
        out.write('| ' + ' | '.join(headers) + ' |\n')
        out.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in rows:
            out.write('| ' + ' | '.join(row) + ' |\n')
        return
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = '  '

    def fmt_row(cells):
        return sep.join(str(c).ljust(w) for c, w in zip(cells, widths))

    out.write(fmt_row(headers) + '\n')
    out.write(sep.join('-' * w for w in widths) + '\n')
    for row in rows:
        out.write(fmt_row(row) + '\n')


def run_coco(
    by_ds: Dict[str, Dict[str, float]],
    metric_names: List[str],
    meta: Dict,
    args,
    path: Path,
    out: TextIO,
    ckpt: Optional[str] = None,
):
    out.write(f'File: {path}\n')
    if ckpt:
        out.write(f'Checkpoint (load_from): {ckpt}\n')
    out.write(f'Mode: coco  |  Datasets: {len(by_ds)}  |  Metrics: {len(metric_names)}\n')
    if meta:
        out.write('Other fields: ' + ', '.join(f'{k}={v}' for k, v in meta.items()) + '\n')
    out.write('\n')

    summary_headers = ['statistic', 'mean', 'n_used']
    if args.ignore_invalid:
        summary_headers.append('n_skipped(-1)')
    summary_rows = []
    for name in metric_names:
        vals = [by_ds[ds][name] for ds in by_ds if name in by_ds[ds]]
        n_all = len(vals)
        if args.ignore_invalid:
            used = [x for x in vals if x != -1.0]
            m = mean(vals, ignore_invalid=True)
            summary_rows.append([
                name, fmt_cell(m), str(len(used)), str(n_all - len(used)),
            ])
        else:
            m = mean(vals, ignore_invalid=False)
            summary_rows.append([name, fmt_cell(m), str(n_all)])

    out.write('=== Summary: mean over datasets (per statistic) ===\n')
    print_table(summary_headers, summary_rows, args.format, out=out)
    out.write('\n')

    if args.no_per_dataset:
        return

    ds_sorted = sorted(by_ds.keys())
    per_headers = ['dataset'] + metric_names
    per_rows = []
    for ds in ds_sorted:
        row = [ds]
        for mn in metric_names:
            v = by_ds[ds].get(mn)
            if v is None:
                row.append('—')
            elif args.ignore_invalid and v == -1.0:
                row.append('n/a')
            else:
                row.append(fmt_cell(v))
        per_rows.append(row)

    out.write(
        '=== Per-dataset values (rows = dataset, columns = metrics) ===\n')
    print_table(per_headers, per_rows, args.format, out=out)
    out.write('\n')

    if 'bbox_mAP' in metric_names:
        m = mean([by_ds[ds]['bbox_mAP'] for ds in ds_sorted], args.ignore_invalid)
        note = 'excluding -1' if args.ignore_invalid else 'all values included'
        out.write(f'macro avg bbox_mAP: {fmt_cell(m)}  ({note})\n')


def run_lvis(
    by_ds: Dict[str, Dict[str, float]],
    metric_names: List[str],
    meta: Dict,
    args,
    path: Path,
    out: TextIO,
    ckpt: Optional[str] = None,
):
    out.write(f'File: {path}\n')
    if ckpt:
        out.write(f'Checkpoint (load_from): {ckpt}\n')
    out.write(f'Mode: lvis  |  Groups: {len(by_ds)}  |  Metrics: {len(metric_names)}\n')
    if meta:
        out.write('Other fields: ' + ', '.join(f'{k}={v}' for k, v in meta.items()) + '\n')
    out.write('\n')

    summary_headers = ['statistic', 'mean', 'n_used']
    if args.ignore_invalid:
        summary_headers.append('n_skipped(-1)')
    summary_rows = []
    for name in metric_names:
        vals = [by_ds[ds][name] for ds in by_ds if name in by_ds[ds]]
        n_all = len(vals)
        if args.ignore_invalid:
            used = [x for x in vals if x != -1.0]
            m = mean(vals, ignore_invalid=True)
            summary_rows.append([
                name, fmt_cell(m), str(len(used)), str(n_all - len(used)),
            ])
        else:
            m = mean(vals, ignore_invalid=False)
            summary_rows.append([name, fmt_cell(m), str(n_all)])

    out.write('=== Summary: mean over LVIS groups (per statistic) ===\n')
    print_table(summary_headers, summary_rows, args.format, out=out)
    out.write('\n')

    if args.no_per_dataset:
        return

    ds_sorted = sorted(by_ds.keys())
    per_headers = ['group'] + metric_names
    per_rows = []
    for ds in ds_sorted:
        row = [ds]
        for mn in metric_names:
            v = by_ds[ds].get(mn)
            if v is None:
                row.append('—')
            elif args.ignore_invalid and v == -1.0:
                row.append('n/a')
            else:
                row.append(fmt_cell(v))
        per_rows.append(row)

    out.write(
        '=== Per-group values (rows = lvis_* block, columns = metrics) ===\n')
    print_table(per_headers, per_rows, args.format, out=out)
    out.write('\n')

    if 'AP' in metric_names:
        m = mean([by_ds[ds]['AP'] for ds in ds_sorted], args.ignore_invalid)
        note = 'excluding -1' if args.ignore_invalid else 'all values included'
        out.write(f'macro avg AP: {fmt_cell(m)}  ({note})\n')


def run_refexp(
    by_ds: Dict[str, Dict[str, float]],
    metric_names: List[str],
    meta: Dict,
    args,
    path: Path,
    out: TextIO,
    ckpt: Optional[str] = None,
):
    ignore_zero = True
    if args.no_ignore_zero:
        ignore_zero = False
    elif args.ignore_zero is not None:
        ignore_zero = args.ignore_zero

    exclude = set()
    if not args.include_val_aggregate:
        exclude.add('val')

    metric_names = refexp_metrics_for_tables(metric_names)

    out.write(f'File: {path}\n')
    if ckpt:
        out.write(f'Checkpoint (load_from): {ckpt}\n')
    out.write(f'Mode: refexp  |  Splits (all keys): {len(by_ds)}  |  '
              f'Metrics (precision@* only): {len(metric_names)}\n')
    out.write(
        'Keys: ``val/refexp/...`` = concat val (also fills missing ``*_val`` '
        'rows in Summary); ``refcoco_testA/...`` = RefCOCO testA; '
        '``refcoco+_*`` / ``refcocog_*`` = that benchmark’s splits.\n')
    out.write(f'Excluded from avg: {sorted(exclude) or "(none)"}  |  '
              f'ignore_zero={ignore_zero}\n')
    if meta:
        out.write('Other fields: ' + ', '.join(f'{k}={v}' for k, v in meta.items()) + '\n')
    out.write('\n')

    ds_for_avg = sorted(s for s in by_ds if s not in exclude)

    summary_headers = ['statistic', 'mean', 'n_used']
    if ignore_zero:
        summary_headers.append('n_skipped(0)')
    summary_rows = []
    for name in metric_names:
        bench = refexp_bench_from_metric(name)
        if bench is not None:
            splits = REFEXP_STATS_SPLITS[bench]
            m, n_used, n_skip0 = refexp_mean_over_mandated_splits(
                by_ds, name, splits, ignore_zero)
            if ignore_zero:
                summary_rows.append([
                    refexp_display_name(name),
                    fmt_cell(m),
                    str(n_used),
                    str(n_skip0),
                ])
            else:
                summary_rows.append([
                    refexp_display_name(name), fmt_cell(m), str(n_used)])
        else:
            vals = [by_ds[ds][name] for ds in ds_for_avg if name in by_ds[ds]]
            raw_n = len(vals)
            if ignore_zero:
                used = [x for x in vals if x != 0.0]
                skipped = raw_n - len(used)
                m = sum(used) / len(used) if used else None
                summary_rows.append([
                    refexp_display_name(name),
                    fmt_cell(m),
                    str(len(used)),
                    str(skipped),
                ])
            else:
                m = sum(vals) / len(vals) if vals else None
                summary_rows.append([refexp_display_name(name), fmt_cell(m), str(raw_n)])

    out.write(
        '=== Summary: mean over mandated splits (RefCOCO/+/g: val+testA+testB or '
        'val+test; mean_precision omitted; zeros ignored) ===\n')
    print_table(summary_headers, summary_rows, args.format, out=out)
    out.write('\n')

    # --- precision@1 per benchmark: mean over splits (metric key: {prefix}_precision@1)
    p1_headers = ['benchmark', 'metric_key', 'mean', 'n_used']
    if ignore_zero:
        p1_headers.append('n_skipped(0)')
    p1_rows = []
    p1_means: List[float] = []
    for label, prefix in REFEXP_BENCHMARKS:
        mkey = f'{prefix}_precision@1'
        splits = REFEXP_STATS_SPLITS[prefix]
        m, n_used, n_skip0 = refexp_mean_over_mandated_splits(
            by_ds, mkey, splits, ignore_zero)
        if ignore_zero:
            p1_rows.append([
                label,
                mkey,
                fmt_cell(m),
                str(n_used),
                str(n_skip0),
            ])
        else:
            p1_rows.append([label, mkey, fmt_cell(m), str(n_used)])
        if m is not None:
            p1_means.append(float(m))
    macro_avg_p1: Optional[float] = (
        sum(p1_means) / len(p1_means) if p1_means else None)
    out.write(
        '=== Precision@1 per benchmark (mean over mandated splits; '
        '``val/`` concat excluded; zeros ignored) ===\n')
    print_table(p1_headers, p1_rows, args.format, out=out)
    out.write('\n')

    def _write_macro_p1() -> None:
        if macro_avg_p1 is None:
            return
        out.write(
            f'macro avg precision@1: {fmt_cell(macro_avg_p1)}  '
            f'(mean of {len(p1_means)} benchmarks over mandated splits)\n')

    if args.no_per_dataset:
        _write_macro_p1()
        return

    ds_sorted = sorted(by_ds.keys())
    per_headers = ['dataset'] + [refexp_display_name(m) for m in metric_names]
    per_rows = []
    for ds in ds_sorted:
        row = [ds]
        for mn in metric_names:
            v = by_ds[ds].get(mn)
            if v is None:
                row.append('—')
            elif ignore_zero and v == 0.0:
                row.append('—')
            else:
                row.append(fmt_cell(v))
        per_rows.append(row)

    out.write(
        '=== Per-split values (rows = split prefix, columns = precision@*; '
        'mean_precision omitted; 0 → — when ignore-zero) ===\n')
    print_table(per_headers, per_rows, args.format, out=out)
    out.write('\n')
    _write_macro_p1()


def _resolve_mode(
    args,
    coco_flat: Dict[str, float],
    refexp_flat: Dict[str, float],
    lvis_flat: Dict[str, float],
) -> str:
    mode = args.mode
    if mode != 'auto':
        return mode
    has_c = bool(coco_flat)
    has_r = bool(refexp_flat)
    has_l = bool(lvis_flat)
    if has_c + has_r + has_l > 1:
        print(
            'Ambiguous JSON: multiple of coco / refexp / lvis key families; '
            'use --mode coco|refexp|lvis',
            file=sys.stderr,
        )
        sys.exit(1)
    if has_r:
        return 'refexp'
    if has_c:
        return 'coco'
    if has_l:
        return 'lvis'
    print(
        'No supported numeric keys: need */coco/* or coco/*, '
        '*/refexp/*, or lvis*/...',
        file=sys.stderr,
    )
    sys.exit(1)


def process_one_json(path: Path, args, out: TextIO) -> None:
    """Load ``path``, detect mode, write tables to ``out``."""
    log_path = path.with_suffix('.log')
    ckpt = parse_load_from_log(log_path)
    eval_res = parse_eval_resolution_from_log(log_path)
    coco_flat, refexp_flat, lvis_flat, meta = load_metrics(path)
    mode = _resolve_mode(args, coco_flat, refexp_flat, lvis_flat)

    if mode == 'coco':
        by_ds, metric_names = pivot_flat(coco_flat, COCO_KEY)
    elif mode == 'refexp':
        by_ds, metric_names = pivot_flat(refexp_flat, REFEXP_KEY)
    else:
        by_ds, metric_names = pivot_lvis_flat(lvis_flat)

    if eval_res:
        out.write(f'Eval resolution (test_dataloader in .log): {eval_res}\n')

    if mode == 'coco':
        run_coco(by_ds, metric_names, meta, args, path, out, ckpt=ckpt)
    elif mode == 'refexp':
        run_refexp(by_ds, metric_names, meta, args, path, out, ckpt=ckpt)
    else:
        run_lvis(by_ds, metric_names, meta, args, path, out, ckpt=ckpt)


def _summary_from_json(path: Path) -> Dict[str, Any]:
    """Compact summary for cross-benchmark comparison."""
    log_path = path.with_suffix('.log')
    load_from = parse_load_from_log(log_path)
    mode = parse_mode_from_log(log_path) or 'text_only'
    decoder_early_exit_layer = parse_decoder_early_exit_layer_from_log(log_path)
    resolution = parse_eval_resolution_from_log(log_path)
    coco_flat, refexp_flat, lvis_flat, _meta = load_metrics(path)

    family = _resolve_mode(
        argparse.Namespace(mode='auto'),
        coco_flat,
        refexp_flat,
        lvis_flat,
    )
    summary: Dict[str, Optional[float]] = {}

    if family == 'coco':
        by_ds, metric_names = pivot_flat(coco_flat, COCO_KEY)
        for name in metric_names:
            vals = [by_ds[ds][name] for ds in by_ds if name in by_ds[ds]]
            summary[f'mean/{name}'] = mean(vals, ignore_invalid=True)
    elif family == 'lvis':
        by_ds, metric_names = pivot_lvis_flat(lvis_flat)
        for name in metric_names:
            vals = [by_ds[ds][name] for ds in by_ds if name in by_ds[ds]]
            summary[f'mean/{name}'] = mean(vals, ignore_invalid=True)
    else:
        by_ds, metric_names = pivot_flat(refexp_flat, REFEXP_KEY)
        for _label, prefix in REFEXP_BENCHMARKS:
            mkey = f'{prefix}_precision@1'
            m, _n, _skip = refexp_mean_over_mandated_splits(
                by_ds, mkey, REFEXP_STATS_SPLITS[prefix], ignore_zero=True)
            summary[f'{prefix}/precision@1'] = m
        p1_vals = [v for k, v in summary.items() if k.endswith('/precision@1') and v is not None]
        summary['macro/precision@1'] = (
            (sum(p1_vals) / len(p1_vals)) if p1_vals else None
        )
        # keep extra means for full comparison view
        metric_names = refexp_metrics_for_tables(metric_names)
        for name in metric_names:
            bench = refexp_bench_from_metric(name)
            if bench is None:
                continue
            m, _n, _skip = refexp_mean_over_mandated_splits(
                by_ds, name, REFEXP_STATS_SPLITS[bench], ignore_zero=True)
            summary[f'mean/{name}'] = m

    try:
        source_json_mtime = float(path.stat().st_mtime)
    except OSError:
        source_json_mtime = 0.0
    try:
        log_p = path.with_suffix('.log')
        source_log_mtime = (
            float(log_p.stat().st_mtime) if log_p.is_file() else 0.0
        )
    except OSError:
        source_log_mtime = 0.0

    return {
        'benchmark': bmk_from_json_path(path),
        'mode': mode,
        'decoder_early_exit_layer': decoder_early_exit_layer,
        'resolution': resolution,
        'load_from': load_from,
        'ckpt_key': ckpt_slot_key(load_from),
        'json_path': str(path),
        'timestamp': path.parent.name,
        'source_json_mtime': source_json_mtime,
        'source_log_mtime': source_log_mtime,
        'metrics': summary,
    }



def _build_metrics_index(
    metrics_root: Path,
) -> Tuple[float, List[Path], Dict[str, List[Path]], List[Path]]:
    metrics_root = metrics_root.resolve()
    catalog: List[Path] = []
    by_train_dir: Dict[str, List[Path]] = defaultdict(list)
    no_train_dir: List[Path] = []
    version = 0.0
    for p in sorted(metrics_root.rglob('*.json')):
        if not is_raw_metrics_json(p):
            continue
        try:
            version = max(version, p.stat().st_mtime)
        except OSError:
            pass
        catalog.append(p)
        log_p = p.with_suffix('.log')
        load_from = parse_load_from_log(log_p)
        train_dir = parse_train_dir_from_load_from(load_from or '')
        if train_dir:
            by_train_dir[str(train_dir).strip()].append(p)
        else:
            no_train_dir.append(p)
    return version, catalog, dict(by_train_dir), no_train_dir


def _raw_metrics_json_manifest(
    metrics_root: Path,
) -> Tuple[float, int]:
    """Max mtime and count of raw metrics JSON files (no log parsing)."""
    metrics_root = metrics_root.resolve()
    version = 0.0
    count = 0
    for p in metrics_root.rglob('*.json'):
        if not is_raw_metrics_json(p):
            continue
        count += 1
        try:
            version = max(version, p.stat().st_mtime)
        except OSError:
            pass
    return version, count


def metrics_tree_max_mtime(metrics_root: Path) -> float:
    """Latest mtime among raw metrics JSON files (no log parsing)."""
    return _raw_metrics_json_manifest(metrics_root.resolve())[0]


def get_metrics_index(
    metrics_root: Path,
) -> Tuple[float, List[Path], Dict[str, List[Path]], List[Path]]:
    """Catalog of raw metrics JSON paths; persisted under ``summary/_storage_cache``."""
    metrics_root = metrics_root.resolve()
    fresh_mt, fresh_count = _raw_metrics_json_manifest(metrics_root)
    cache_path = _metrics_index_cache_path(metrics_root)
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and float(payload.get("version", 0.0)) >= fresh_mt - 1e-6
                and int(payload.get("catalog_count", -1)) == fresh_count
            ):
                return _deserialize_metrics_index(payload)
        except Exception:
            pass
    built = _build_metrics_index(metrics_root)
    try:
        with _METRICS_INDEX_LOCK:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(_serialize_metrics_index(built), ensure_ascii=False),
                encoding="utf-8",
            )
    except OSError:
        pass
    return built


def match_metrics_json_for_experiment(
    metrics_root: Path,
    experiment_name: str,
) -> List[Path]:
    """Return raw metrics JSON paths belonging to one training experiment."""
    _version, _catalog, by_train_dir, no_train_dir = get_metrics_index(metrics_root)
    exp_key = experiment_name.strip()
    matched: List[Path] = []
    seen: set[Path] = set()
    for train_dir, paths in by_train_dir.items():
        if experiment_dir_names_match(exp_key, train_dir):
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    matched.append(p)
    for p in no_train_dir:
        if exp_key in p.parts and p not in seen:
            seen.add(p)
            matched.append(p)
    return matched


def max_mtime_metrics_json_for_experiment(
    metrics_root: Path,
    experiment_name: str,
) -> float:
    mt = 0.0
    for p in match_metrics_json_for_experiment(metrics_root, experiment_name):
        try:
            mt = max(mt, p.stat().st_mtime)
        except OSError:
            pass
    return mt


    return mt


def _resolved_json_path_key(path: Path | str) -> str:
    """Site-agnostic key: ``metrics/<bmk>/.../*.json`` under work_dirs."""
    raw = str(path).replace('\\', '/')

    def _metrics_relative(s: str) -> str | None:
        anchor = '/work_dirs/metrics/'
        idx = s.find(anchor)
        if idx >= 0:
            return s[idx + len('/work_dirs/'):]
        parts = Path(s).parts
        for i, part in enumerate(parts):
            if part == 'metrics':
                return str(Path(*parts[i:]))
        return None

    try:
        rel = _metrics_relative(str(Path(path).resolve()).replace('\\', '/'))
        if rel:
            return rel
    except OSError:
        pass
    rel = _metrics_relative(raw)
    if rel:
        return rel
    return raw


def _run_source_stale(run: Dict[str, Any], path: Path) -> bool:
    """True when raw eval JSON or sidecar log is newer than cached run metadata."""
    try:
        json_mt = float(path.stat().st_mtime)
    except OSError:
        return True
    try:
        log_p = path.with_suffix('.log')
        log_mt = float(log_p.stat().st_mtime) if log_p.is_file() else 0.0
    except OSError:
        log_mt = 0.0
    cached_json = run.get('source_json_mtime')
    if cached_json is None:
        cached_json = run.get('report_mtime')
    try:
        cached_json_f = float(cached_json or 0.0)
    except (TypeError, ValueError):
        cached_json_f = 0.0
    try:
        cached_log_f = float(run.get('source_log_mtime') or 0.0)
    except (TypeError, ValueError):
        cached_log_f = 0.0
    if cached_json_f <= 0.0:
        return True
    if json_mt > cached_json_f + 1e-6:
        return True
    if log_mt > cached_log_f + 1e-6:
        return True
    return False


def attach_eval_sources_fingerprint(
    metrics_root: Path,
    experiment_name: str,
    payload: Dict[str, Any],
) -> None:
    """Store path list + max mtime so refresh can skip per-file stat when unchanged."""
    paths = match_metrics_json_for_experiment(metrics_root, experiment_name)
    keys = sorted(_resolved_json_path_key(p) for p in paths)
    max_mt = 0.0
    for p in paths:
        try:
            max_mt = max(max_mt, float(p.stat().st_mtime))
        except OSError:
            pass
    payload['_eval_sources_count'] = len(keys)
    payload['_eval_sources_max_mtime'] = max_mt
    payload['_eval_sources_paths'] = keys


def eval_metrics_sources_unchanged(
    metrics_root: Path,
    experiment_name: str,
    payload: Dict[str, Any],
) -> bool:
    """True when persisted eval runs still cover the same raw JSON paths and mtimes."""
    eval_runs = [
        r for r in (payload.get('runs') or [])
        if str(r.get('source', '')) != 'training_log' and r.get('json_path')
    ]
    cached_paths = payload.get('_eval_sources_paths')
    if isinstance(cached_paths, list) and cached_paths:
        matched = {
            _resolved_json_path_key(p)
            for p in match_metrics_json_for_experiment(metrics_root, experiment_name)
        }
        if matched != {_resolved_json_path_key(x) for x in cached_paths}:
            return False
        try:
            cached_max = float(payload.get('_eval_sources_max_mtime') or 0.0)
        except (TypeError, ValueError):
            cached_max = 0.0
        if cached_max > 0.0:
            fresh_max = max_mtime_metrics_json_for_experiment(metrics_root, experiment_name)
            if fresh_max <= cached_max + 1e-6:
                return True
    matched = {
        _resolved_json_path_key(p)
        for p in match_metrics_json_for_experiment(metrics_root, experiment_name)
    }
    existing = {
        _resolved_json_path_key(str(r.get('json_path')))
        for r in eval_runs
    }
    if matched != existing:
        return False
    for run in eval_runs:
        jp = Path(str(run.get('json_path')))
        if not jp.is_file():
            return False
        if _run_source_stale(run, jp):
            return False
    return True


def collect_experiment_metrics_from_root(
    metrics_root: Path,
    experiment_name: str,
    *,
    latest_per_benchmark: bool = True,
    existing_runs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect benchmark summaries for one experiment from metrics root."""
    metrics_root = metrics_root.resolve()
    matched = match_metrics_json_for_experiment(metrics_root, experiment_name)
    cached_by_path: Dict[str, Dict[str, Any]] = {}
    if existing_runs:
        for run in existing_runs:
            if str(run.get('source', '')) == 'training_log':
                continue
            jp = run.get('json_path')
            if not jp:
                continue
            cached_by_path[_resolved_json_path_key(str(jp))] = dict(run)

    runs: List[Dict[str, Any]] = []
    parsed = 0
    reused = 0
    for p in matched:
        key = _resolved_json_path_key(p)
        cached = cached_by_path.get(key)
        if cached is not None and not _run_source_stale(cached, p):
            runs.append(dict(cached))
            reused += 1
            continue
        try:
            runs.append(_summary_from_json(p))
            parsed += 1
        except (ValueError, json.JSONDecodeError, OSError):
            # skip non-metrics or corrupted files quietly in experiment mode
            continue
    if latest_per_benchmark:
        latest: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        for item in runs:
            decx = item.get('decoder_early_exit_layer')
            decx_key = f"decx{decx}" if decx is not None else 'decxNA'
            res = item.get('resolution')
            res_key = str(res) if res else 'resNA'
            lf = item.get('load_from')
            ck_key = ckpt_slot_key(lf if isinstance(lf, str) else None)
            key = (str(item['benchmark']), str(item['mode']), decx_key, res_key, ck_key)
            cur = latest.get(key)
            if cur is None or str(item['timestamp']) > str(cur['timestamp']):
                latest[key] = item
        runs = sorted(
            latest.values(),
            key=lambda x: (x['benchmark'], x['mode'], str(x.get('resolution') or '')),
        )
    return {
        'experiment_name': experiment_name,
        'metrics_root': str(metrics_root),
        'num_runs': len(runs),
        'runs': runs,
        'metrics_parse_reused': reused,
        'metrics_parse_parsed': parsed,
    }


def main():
    args = parse_args()
    if args.metrics_root:
        print(
            '--metrics-root is deprecated; use --scan-root instead',
            file=sys.stderr,
        )
        if not args.scan_root:
            args.scan_root = args.metrics_root

    if args.experiment_name:
        if args.json_path:
            print(
                '--experiment-name is mutually exclusive with json_path',
                file=sys.stderr,
            )
            sys.exit(1)
        root = Path(args.scan_root or 'work_dirs/metrics')
        result = collect_experiment_metrics_from_root(
            root,
            args.experiment_name,
            latest_per_benchmark=True,
        )

        if args.dump_json:
            output_text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
            default_path = (
                root / 'summary' / sanitize_mode_for_filename(args.experiment_name)
                / f"{sanitize_mode_for_filename(args.experiment_name)}_metrics.json"
            )
        else:
            if args.format == 'markdown':
                # per-dataset markdowns, same style as metrics_260316_all/*
                out_dir = root / 'summary' / sanitize_mode_for_filename(args.experiment_name)
                if args.out_file:
                    out_dir = Path(args.out_file)
                out_dir.mkdir(parents=True, exist_ok=True)

                failed: List[str] = []
                written: List[Path] = []
                for run in result['runs']:
                    json_path = Path(str(run.get('json_path', '')))
                    if not json_path.is_file():
                        continue
                    outp = batch_report_path(
                        json_path,
                        root,
                        out_dir,
                        'markdown',
                        legacy_long_path=False,
                    )
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(outp, 'w', encoding='utf-8') as out:
                            process_one_json(json_path, args, out)
                        written.append(outp)
                    except (json.JSONDecodeError, OSError, ValueError) as exc:
                        failed.append(f"{json_path}: {exc}")

                # also write one short index for quick navigation
                index_path = out_dir / f"{sanitize_mode_for_filename(args.experiment_name)}_index.md"
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Experiment Metrics Index: {args.experiment_name}\n\n")
                    f.write(f"- matched runs: {result['num_runs']}\n")
                    f.write(f"- generated md files: {len(written)}\n\n")
                    for p in sorted(written):
                        f.write(f"- `{p.name}`\n")
                    if failed:
                        f.write("\n## Failed\n")
                        for item in failed:
                            f.write(f"- {item}\n")
                print(f'Write {out_dir} (files={len(written)})')
                return
            else:
                lines = [
                    f"Experiment: {result['experiment_name']}  |  matched runs: {result['num_runs']}"
                ]
                for run in result['runs']:
                    metric_preview = ', '.join(
                        f'{k}={fmt_cell(v)}' for k, v in list(run['metrics'].items())[:6]
                    )
                    lines.append(
                        f"- {run['benchmark']} ({run['mode']}) @ {run['timestamp']}: "
                        f"{metric_preview}"
                    )
                output_text = '\n'.join(lines) + '\n'
                default_path = (
                    root / 'summary' / sanitize_mode_for_filename(args.experiment_name)
                    / f"{sanitize_mode_for_filename(args.experiment_name)}_metrics.txt"
                )

        outp = Path(args.out_file) if args.out_file else default_path
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(output_text, encoding='utf-8')
        print(f'Write {outp}')
        return

    if args.scan_root:
        if not args.name_substr or not args.out_dir:
            print(
                '--scan-root requires --name-substr and --out-dir',
                file=sys.stderr,
            )
            sys.exit(1)
        if args.json_path:
            print(
                'Do not pass json_path together with --scan-root',
                file=sys.stderr,
            )
            sys.exit(1)
        if args.out_file:
            print(
                'Do not use -o/--out-file with --scan-root (use --out-dir)',
                file=sys.stderr,
            )
            sys.exit(1)
        root = Path(args.scan_root)
        if not root.is_dir():
            print(f'Not a directory: {root}', file=sys.stderr)
            sys.exit(1)
        paths = discover_json_files(root, args.name_substr)
        if args.latest_per_benchmark:
            paths = latest_json_per_benchmark(paths)
        if not paths:
            print(
                f'No *.json under {root} with {args.name_substr!r} in filename',
                file=sys.stderr,
            )
            sys.exit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        failed: List[Path] = []
        for path in paths:
            outp = batch_report_path(
                path,
                root,
                out_dir,
                args.format,
                legacy_long_path=args.legacy_batch_names,
            )
            outp.parent.mkdir(parents=True, exist_ok=True)
            print(f'Write {outp}  <=  {path}', file=sys.stderr)
            try:
                with open(outp, 'w', encoding='utf-8') as out:
                    process_one_json(path, args, out)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                print(f'ERROR (skip): {path}\n  {exc}', file=sys.stderr)
                failed.append(path)
                try:
                    outp.unlink()
                except OSError:
                    pass
        if failed:
            print(
                f'Finished with {len(failed)} failure(s) out of {len(paths)} file(s).',
                file=sys.stderr,
            )
            sys.exit(1)
        return

    if not args.json_path:
        print(
            'Provide a metrics .json path, or use '
            '--scan-root --name-substr --out-dir, or --experiment-name',
            file=sys.stderr,
        )
        sys.exit(1)

    path = Path(args.json_path)
    if not path.is_file():
        print(f'File not found: {path}', file=sys.stderr)
        sys.exit(1)

    if args.out_file:
        outp = out_path_for_format(args.out_file, args.format)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as out:
            process_one_json(path, args, out)
    else:
        process_one_json(path, args, sys.stdout)


if __name__ == '__main__':
    main()
