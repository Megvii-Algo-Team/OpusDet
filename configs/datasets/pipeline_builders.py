from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple

OD_META_KEYS = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
    'flip', 'flip_direction', 'text', 'custom_entities')

ODVG_META_KEYS = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
    'flip', 'flip_direction', 'text', 'custom_entities', 'tokens_positive',
    'dataset_mode')

TEST_META_KEYS = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text',
    'custom_entities', 'tokens_positive', 'mode', 'prompt_path',
    'embed_label_map')

DEFAULT_MULTI_SCALES = [
    (480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333),
    (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333),
    (800, 1333)
]
DEFAULT_CROP_RESIZE_SCALES = [(400, 4200), (500, 4200), (600, 4200)]


def build_multiscale_od_pipeline(
        *,
        load_image_kwargs: Optional[Dict] = None,
        with_random_flip: bool = True,
        with_filter: bool = True,
        filter_min_gt_bbox_wh: Tuple[float, float] = (1e-2, 1e-2),
        with_text_sampling: bool = False,
        tokenizer_name: Optional[str] = None,
        num_sample_negative: int = 85,
        label_map_file: Optional[str] = None,
        max_tokens: int = 256):
    """Build RT-DETR style multi-scale OD/ODVG train pipeline."""
    pipeline = [
        dict(type='LoadImageFromFile', **(load_image_kwargs or {})),
        dict(type='LoadAnnotations', with_bbox=True),
    ]
    if with_random_flip:
        pipeline.append(dict(type='RandomFlip', prob=0.5))

    pipeline.append(
        dict(
            type='RandomChoice',
            transforms=[[
                dict(
                    type='RandomChoiceResize',
                    scales=deepcopy(DEFAULT_MULTI_SCALES),
                    keep_ratio=True)
            ], [
                dict(
                    type='RandomChoiceResize',
                    scales=deepcopy(DEFAULT_CROP_RESIZE_SCALES),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=deepcopy(DEFAULT_MULTI_SCALES),
                    keep_ratio=True)
            ]]))

    if with_filter:
        pipeline.append(
            dict(type='FilterAnnotations',
                 min_gt_bbox_wh=filter_min_gt_bbox_wh))

    if with_text_sampling:
        pipeline.append(
            dict(
                type='RandomSamplingNegPos',
                tokenizer_name=tokenizer_name,
                num_sample_negative=num_sample_negative,
                label_map_file=label_map_file,
                max_tokens=max_tokens))

    pipeline.append(
        dict(
            type='PackDetInputs',
            meta_keys=ODVG_META_KEYS if with_text_sampling else OD_META_KEYS))
    return pipeline

def _dedupe_square_sizes(
        head: Tuple[int, int],
        rest: Sequence[Tuple[int, int]],
) -> list:
    """``head`` first, then ``rest``, dropping duplicate ``(h, w)``."""
    out: list = []
    seen = set()
    for s in (head,) + tuple(rest):
        t = tuple(int(x) for x in s)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _fixed_scale_random_choice_transforms(
        image_size: Tuple[int, int],
        multi_scales: Optional[Sequence[Tuple[int, int]]],
        *,
        with_mosaic: bool,
        weak_ratio_range: Tuple[float, float] = (0.5, 1.5),
        mosaic_ratio_range: Tuple[float, float] = (1.0, 2.0),
) -> list:
    """Inner ``RandomChoice`` branches for :func:`build_fixed_scale_od_pipeline`.

    ``image_size`` is ``(h, w)`` for branch 2–3 (``RandomResize`` / crop / ``Pad``
    / ``CachedMosaic``); mosaic ``img_scale`` is ``(w, h)``.

    Branch 1 (``Resize`` + ``Pad``): if ``multi_scales`` is empty, only
    ``image_size``; otherwise a nested ``RandomChoice`` over
    ``image_size`` merged with ``multi_scales`` (deduped, ``image_size`` first).
    """
    canonical = tuple(int(x) for x in image_size)
    extras = [tuple(int(x) for x in s) for s in (multi_scales or ())]
    pad_candidates = _dedupe_square_sizes(canonical, extras)
    if len(pad_candidates) == 1:
        s0 = pad_candidates[0]
        branch_resize_pad = [
            dict(type='Resize', scale=s0, keep_ratio=True),
            dict(
                type='Pad',
                size=s0,
                pad_val=dict(img=(114, 114, 114))),
        ]
    else:
        branch_resize_pad = [
            dict(
                type='RandomChoice',
                transforms=[
                    [
                        dict(type='Resize', scale=s0, keep_ratio=True),
                        dict(
                            type='Pad',
                            size=s0,
                            pad_val=dict(img=(114, 114, 114))),
                    ]
                    for s0 in pad_candidates
                ],
            )
        ]

    transforms = [
        branch_resize_pad,
        [
            dict(
                type='RandomResize',
                scale=canonical,
                ratio_range=weak_ratio_range,
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=canonical,
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='Pad',
                size=canonical,
                pad_val=dict(img=(114, 114, 114))),
        ],
    ]
    if with_mosaic:
        transforms.append([
            dict(
                type='CachedMosaic',
                img_scale=(int(canonical[1]), int(canonical[0])),
                pad_val=114.0,
            ),
            dict(
                type='RandomResize',
                scale=canonical,
                ratio_range=mosaic_ratio_range,
                keep_ratio=True,
            ),
            dict(
                type='RandomCrop',
                crop_size=canonical,
                recompute_bbox=True,
                allow_negative_crop=True,
            ),
            dict(
                type='Pad',
                size=canonical,
                pad_val=dict(img=(114, 114, 114))),
        ])
    return transforms

def build_fixed_scale_od_pipeline(
        *,
        image_size: Tuple[int, int],
        multi_scales: Optional[Sequence[Tuple[int, int]]] = None,
        load_image_kwargs: Optional[Dict] = None,
        with_mosaic: bool = True,
        with_random_flip: bool = True,
        with_filter: bool = True,
        filter_min_gt_bbox_wh: Tuple[float, float] = (1e-2, 1e-2),
        with_hsv_random_aug: bool = True,
        with_text_sampling: bool = False,
        tokenizer_name: Optional[str] = None,
        num_sample_negative: int = 85,
        label_map_file: Optional[str] = None,
        max_tokens: int = 256,
        weak_ratio_range: Tuple[float, float] = (0.5, 1.5),
        mosaic_ratio_range: Tuple[float, float] = (1.0, 2.0),
):
    """Build fixed-size OD/ODVG train pipeline.

    ``image_size`` is the main ``(h, w)`` square for branch 2–3 (random resize,
    crop, pad, mosaic chain). Optional ``multi_scales`` lists **extra** squares
    for branch 1 only (``Resize`` + ``Pad``); branch 1 then ``RandomChoice``\ s
    over ``image_size`` plus those extras (deduped). Omit ``multi_scales`` to
    keep branch 1 fixed at ``image_size`` only.
    """
    pipeline = [
        dict(type='LoadImageFromFile', **(load_image_kwargs or {})),
        dict(type='LoadAnnotations', with_bbox=True),
    ]

    pipeline.append(
        dict(
            type='RandomChoice',
            transforms=_fixed_scale_random_choice_transforms(
                tuple(int(x) for x in image_size),
                multi_scales,
                with_mosaic=with_mosaic,
                weak_ratio_range=weak_ratio_range,
                mosaic_ratio_range=mosaic_ratio_range,
            ),
        ))
    if with_hsv_random_aug:
        pipeline.append(
            dict(type='YOLOXHSVRandomAug'))
    if with_random_flip:
        pipeline.append(dict(type='RandomFlip', prob=0.5))
    if with_filter:
        pipeline.append(
            dict(type='FilterAnnotations',
                 min_gt_bbox_wh=filter_min_gt_bbox_wh))
    if with_text_sampling:
        pipeline.append(
            dict(
                type='RandomSamplingNegPos',
                tokenizer_name=tokenizer_name,
                num_sample_negative=num_sample_negative,
                label_map_file=label_map_file,
                max_tokens=max_tokens))
    pipeline.append(
        dict(
            type='PackDetInputs',
            meta_keys=ODVG_META_KEYS if with_text_sampling else OD_META_KEYS))
    return pipeline


def build_test_pipeline(*,
                        scale: Tuple[int, int],
                        with_pad: bool = False,
                        pad_size: Optional[Tuple[int, int]] = None):
    """Build OD test/eval pipeline."""
    pipeline = [
        dict(
            type='LoadImageFromFile',
            backend_args=None,
            imdecode_backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='FixScaleResize', scale=scale, keep_ratio=True, backend='pillow'),
    ]
    if with_pad:
        pipeline.append(
            dict(
                type='Pad',
                size=pad_size if pad_size is not None else scale,
                pad_val=dict(img=(114, 114, 114))))
    pipeline.append(dict(type='PackDetInputs', meta_keys=TEST_META_KEYS))
    return pipeline


def inject_odinw35_visual_prompt_pipeline(datasets, dataset_prefixes):
    """ODinW-35 visual.G: see ``odinw35_visual_prompt.inject_odinw35_visual_prompt_pipeline``."""
    from opus.datasets.transforms.odinw35_visual_prompt import (
        inject_odinw35_visual_prompt_pipeline as _inject)
    return _inject(datasets, dataset_prefixes)
