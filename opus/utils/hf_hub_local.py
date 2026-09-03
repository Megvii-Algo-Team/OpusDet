# Copyright (c) OpenMMLab. All rights reserved.
"""Resolve bundled Hugging Face config dirs (no weights) for offline init."""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_CONFIG_ROOT = _REPO_ROOT / 'hf_configs'


def hf_config_root() -> Path:
    """Root directory of bundled ``org/model`` config trees."""
    env = os.environ.get('MMDET_HF_CONFIG_ROOT', '').strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_HF_CONFIG_ROOT


def resolve_hf_hub_path(hub_id: str) -> str:
    """Return a local config directory if bundled, else the original *hub_id*."""
    if not hub_id:
        return hub_id
    if hub_id.startswith(('/', './', '../')):
        return hub_id
    if os.path.isdir(hub_id):
        return hub_id

    local = hf_config_root() / hub_id
    if (local / 'config.json').is_file():
        return str(local)
    return hub_id


def hf_config_source(hub_id: str, use_pretrain: bool) -> str:
    if use_pretrain:
        return hub_id
    return resolve_hf_hub_path(hub_id)
