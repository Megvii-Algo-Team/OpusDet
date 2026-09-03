# Copyright (c) OpenMMLab. All rights reserved.
"""ODinW-35 helpers for visual.G prompt paths and train/test label aliasing."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .dataset_transforms import (
    DEFAULT_VISUAL_PROMPT_PATH_TEMPLATE,
    inject_visual_prompt_path_pipeline,
)

# Same layout as generic visual prompts; named for ODinW-35 configs.
ODINW35_PATH_TEMPLATE = DEFAULT_VISUAL_PROMPT_PATH_TEMPLATE


def infer_odinw35_train_ann_from_valid_ann(valid_ann: str) -> Optional[str]:
    """Guess ODinW train ann next to a valid/test ann path."""
    candidates = []
    if '/valid/new_annotations_without_background.json' in valid_ann:
        candidates.append(
            valid_ann.replace(
                '/valid/new_annotations_without_background.json',
                '/train/annotations_without_background.json'))
    if '/valid/annotations_without_background.json' in valid_ann:
        candidates.append(
            valid_ann.replace(
                '/valid/annotations_without_background.json',
                '/train/annotations_without_background.json'))
    if 'val_annotations_without_background.json' in valid_ann:
        candidates.append(
            valid_ann.replace(
                'val_annotations_without_background.json',
                'train_annotations_without_background.json'))
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def _load_odinw35_train_category_names(train_ann: str) -> List[str]:
    data = json.loads(Path(train_ann).read_text(encoding='utf-8'))
    if isinstance(data, list):
        data = data[0]
    cats = {int(c['id']): str(c['name']) for c in data.get('categories', [])}
    return [cats[cid] for cid in sorted(cats)]


def build_odinw35_embed_label_map(
    test_classes: Sequence[str], train_ann: str
) -> Optional[Dict[str, str]]:
    """Map ODinW metainfo/test class name -> train-ann name for pkl lookup.

    Aligns categories by sorted train category id vs ``metainfo['classes']``
    order (same convention as ODinW-35 eval config).
    """
    test_ordered = [str(x) for x in test_classes]
    train_ordered = _load_odinw35_train_category_names(train_ann)
    if len(train_ordered) != len(test_ordered):
        return None
    mapping = {
        test: train
        for test, train in zip(test_ordered, train_ordered)
        if test != train
    }
    return mapping or None


def inject_odinw35_visual_prompt_pipeline(datasets, dataset_prefixes):
    """ODinW-35 visual.G: per-subset ``prompt_path`` and test-time label map."""
    embed_label_maps = []
    for ds in datasets:
        metainfo = ds.get('metainfo') or {}
        test_classes = metainfo.get('classes') or ()
        train_ann = infer_odinw35_train_ann_from_valid_ann(
            ds.get('ann_file', ''))
        if train_ann and test_classes:
            embed_label_maps.append(
                build_odinw35_embed_label_map(test_classes, train_ann))
        else:
            embed_label_maps.append(None)

    return inject_visual_prompt_path_pipeline(
        datasets,
        dataset_prefixes,
        path_template=ODINW35_PATH_TEMPLATE,
        embed_label_maps=embed_label_maps,
    )
