# Copyright (c) OpenMMLab. All rights reserved.
"""Strip HF/ImageNet pretrain from model cfg before build (checkpoint loaded later)."""
from collections.abc import Mapping
from typing import Optional

from .hf_hub_local import resolve_hf_hub_path

_USE_PRETRAIN_MODULE_TYPES = frozenset({
    'CLIPModel',
    'HFDINOv3ViTBackbone',
    'HFDINOv3ConvNeXtBackbone',
})


def _strip_pretrained_in_model_cfg(node):
    if not isinstance(node, Mapping):
        return
    if 'init_cfg' in node:
        ic = node.get('init_cfg')
        if isinstance(ic, dict) and ic.get('type') == 'Pretrained':
            node['init_cfg'] = None
        elif isinstance(ic, (list, tuple)):
            filtered = [
                x for x in ic
                if not (isinstance(x, dict) and x.get('type') == 'Pretrained')
            ]
            node['init_cfg'] = filtered if filtered else None
    if node.get('pretrained') is not None and isinstance(node.get('pretrained'), str):
        node['pretrained'] = None
    for k, v in node.items():
        if k == 'init_cfg':
            continue
        if isinstance(v, Mapping):
            _strip_pretrained_in_model_cfg(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, Mapping):
                    _strip_pretrained_in_model_cfg(item)


def _disable_use_pretrain_recursively(node):
    if not isinstance(node, Mapping):
        return
    mod_type = node.get('type')
    if 'use_pretrain' in node or mod_type in _USE_PRETRAIN_MODULE_TYPES:
        node['use_pretrain'] = False
        for key in ('model_name', 'name'):
            if key in node and isinstance(node[key], str):
                node[key] = resolve_hf_hub_path(node[key])
    for v in node.values():
        if isinstance(v, Mapping):
            _disable_use_pretrain_recursively(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, Mapping):
                    _disable_use_pretrain_recursively(item)


def _cfg_mapping_root(cfg_obj):
    if hasattr(cfg_obj, '_cfg_dict'):
        return cfg_obj._cfg_dict
    return cfg_obj


def strip_pretrained_for_inference(cfg, logger=None):
    """Skip HF/ImageNet pretrain at build; load checkpoint after build if set."""
    root = _cfg_mapping_root(cfg)
    if 'resume' in root and root.get('resume') is not None:
        root['resume'] = None
        if logger is not None:
            logger.info('strip_pretrained: cleared cfg.resume')
    model_cfg = cfg.get('model', None)
    if model_cfg is None:
        return
    _strip_pretrained_in_model_cfg(model_cfg)
    _disable_use_pretrain_recursively(model_cfg)
    if logger is not None:
        logger.info(
            'strip_pretrained: cleared Pretrained init_cfg; '
            'use_pretrain=False on HF/CLIP submodules under model.')


strip_pretrained_for_flops = strip_pretrained_for_inference
