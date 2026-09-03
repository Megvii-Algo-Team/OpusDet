# Copyright (c) OpenMMLab. All rights reserved.
"""Generic per-sample ``prompt_path`` injection for visual.G evaluation."""
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS

# Default layout used by gen_visual_prompt_embeddings_fast / test_common REGEN.
DEFAULT_VISUAL_PROMPT_PREDS_ROOT = 'work_dirs/visual_prompts'
DEFAULT_CKPT_DIR_ENV = 'VISUAL_PROMPT_CKPT_DIR'
DEFAULT_VISUAL_PROMPT_PATH_TEMPLATE = (
    '{preds_root}/{ckpt_dir}/preds/{prefix}.pkl')


def format_visual_prompt_path(
    path_template: str,
    *,
    prefix: str = '',
    preds_root: str = DEFAULT_VISUAL_PROMPT_PREDS_ROOT,
    ckpt_dir: Optional[str] = None,
    ckpt_dir_env: str = DEFAULT_CKPT_DIR_ENV,
    **template_vars: Any,
) -> Optional[str]:
    """Format ``path_template``; return None if ``{ckpt_dir}`` is required but missing."""
    import os
    ckpt = ckpt_dir if ckpt_dir is not None else os.environ.get(ckpt_dir_env, '')
    if '{ckpt_dir}' in path_template and not ckpt:
        return None
    return path_template.format(
        preds_root=preds_root.rstrip('/'),
        ckpt_dir=ckpt,
        prefix=prefix,
        **template_vars,
    )


def build_set_visual_prompt_path_transform(
    *,
    prefix: str = '',
    prompt_path: Optional[str] = None,
    path_template: Optional[str] = None,
    preds_root: str = DEFAULT_VISUAL_PROMPT_PREDS_ROOT,
    ckpt_dir: Optional[str] = None,
    ckpt_dir_env: str = DEFAULT_CKPT_DIR_ENV,
    embed_label_map: Optional[Mapping[str, str]] = None,
    **template_vars: Any,
) -> dict:
    """Build a pipeline dict for :class:`SetVisualPromptPath`."""
    cfg: Dict[str, Any] = dict(type='SetVisualPromptPath')
    if prompt_path is not None:
        cfg['prompt_path'] = prompt_path
    if path_template is not None:
        cfg['path_template'] = path_template
    if prefix:
        cfg['prefix'] = prefix
    if preds_root != DEFAULT_VISUAL_PROMPT_PREDS_ROOT:
        cfg['preds_root'] = preds_root
    if ckpt_dir is not None:
        cfg['ckpt_dir'] = ckpt_dir
    if ckpt_dir_env != DEFAULT_CKPT_DIR_ENV:
        cfg['ckpt_dir_env'] = ckpt_dir_env
    if embed_label_map:
        cfg['embed_label_map'] = dict(embed_label_map)
    cfg.update(template_vars)
    return cfg


@TRANSFORMS.register_module()
class SetVisualPromptPath(BaseTransform):
    """Write visual.G paths and optional label alias map into ``results``.

    Either a fixed ``prompt_path``, or a ``path_template`` with placeholders:

    - ``{preds_root}``, ``{ckpt_dir}``, ``{prefix}``, plus extra ctor kwargs.

    ``ckpt_dir`` defaults to env ``ckpt_dir_env`` (default ``VISUAL_PROMPT_CKPT_DIR``).

    ``embed_label_map`` maps test/metainfo class names to names stored in the
    visual prompt pkl (used by ODinW-35 where train/test names differ).
    """

    def __init__(
        self,
        prompt_path: Optional[str] = None,
        path_template: Optional[str] = None,
        prefix: str = '',
        preds_root: str = DEFAULT_VISUAL_PROMPT_PREDS_ROOT,
        ckpt_dir: Optional[str] = None,
        ckpt_dir_env: str = DEFAULT_CKPT_DIR_ENV,
        embed_label_map: Optional[Mapping[str, str]] = None,
        **template_vars: Any,
    ) -> None:
        self.prompt_path = prompt_path
        self.path_template = path_template
        self.prefix = prefix
        self.preds_root = preds_root
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir_env = ckpt_dir_env
        self.embed_label_map = dict(embed_label_map) if embed_label_map else None
        self.template_vars = template_vars

    def transform(self, results: dict) -> dict:
        path = self.prompt_path
        if path is None and self.path_template is not None:
            path = format_visual_prompt_path(
                self.path_template,
                prefix=self.prefix,
                preds_root=self.preds_root,
                ckpt_dir=self.ckpt_dir,
                ckpt_dir_env=self.ckpt_dir_env,
                **self.template_vars,
            )
        if path:
            results['prompt_path'] = path
        if self.embed_label_map:
            results['embed_label_map'] = self.embed_label_map
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'prompt_path={self.prompt_path!r}, '
            f'path_template={self.path_template!r}, prefix={self.prefix!r}, '
            f'embed_label_map={self.embed_label_map!r})')


def inject_visual_prompt_path_pipeline(
    datasets: Sequence[dict],
    prefixes: Sequence[str],
    *,
    path_template: str = DEFAULT_VISUAL_PROMPT_PATH_TEMPLATE,
    preds_root: str = DEFAULT_VISUAL_PROMPT_PREDS_ROOT,
    ckpt_dir: Optional[str] = None,
    ckpt_dir_env: str = DEFAULT_CKPT_DIR_ENV,
    embed_label_maps: Optional[Sequence[Optional[Mapping[str, str]]]] = None,
    pack_meta_keys: Optional[Tuple[str, ...]] = None,
    insert_before_pack: bool = True,
) -> List[dict]:
    """Clone each dataset cfg and insert :class:`SetVisualPromptPath` before pack.

    Args:
        datasets: Dataset config dicts (each with ``pipeline``).
        prefixes: One entry per dataset (e.g. ODinW ``dataset_prefix``).
        path_template: Formatted per dataset with ``prefix=<prefixes[i]>``.
        embed_label_maps: Optional per-dataset test->train label alias maps.
        pack_meta_keys: ``PackDetInputs.meta_keys``; default imports
            ``TEST_META_KEYS`` from ``pipeline_builders`` (includes ``prompt_path``).
    """
    if len(datasets) != len(prefixes):
        raise ValueError(
            f'inject_visual_prompt_path_pipeline: len(datasets)='
            f'{len(datasets)} != len(prefixes)={len(prefixes)}')
    if embed_label_maps is not None and len(embed_label_maps) != len(datasets):
        raise ValueError(
            f'inject_visual_prompt_path_pipeline: len(embed_label_maps)='
            f'{len(embed_label_maps)} != len(datasets)={len(datasets)}')

    if pack_meta_keys is None:
        from configs.datasets.pipeline_builders import (
            TEST_META_KEYS)
        pack_meta_keys = TEST_META_KEYS

    out: List[dict] = []
    for i, (ds, prefix) in enumerate(zip(datasets, prefixes)):
        cfg = deepcopy(ds)
        pipeline = list(cfg['pipeline'])
        if pipeline[-1].get('type') == 'PackDetInputs':
            pack = deepcopy(pipeline[-1])
            pack['meta_keys'] = pack_meta_keys
            pipeline[-1] = pack
        label_map = None
        if embed_label_maps is not None:
            label_map = embed_label_maps[i]
        step = build_set_visual_prompt_path_transform(
            prefix=prefix,
            path_template=path_template,
            preds_root=preds_root,
            ckpt_dir=ckpt_dir,
            ckpt_dir_env=ckpt_dir_env,
            embed_label_map=label_map,
        )
        if insert_before_pack:
            pipeline.insert(-1, step)
        else:
            pipeline.append(step)
        cfg['pipeline'] = pipeline
        out.append(cfg)
    return out
