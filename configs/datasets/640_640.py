# Datasets for OPUS — fixed 640×640. MM-GDINO-style data_root.
# Full pretrain: o365, oiv6, v3det, crowdhuman, hiertext, bamboo, sa-1b,
# goldg (flickr30k/gqa), bamboo-cls, cc3m; val: coco2017.
_base_ = [
    '../_base_/datasets/coco_detection.py'
]
import os

from configs.datasets.pipeline_builders import (
    build_fixed_scale_od_pipeline, build_test_pipeline)

# Override anytime: export ROOT_DATA_DIR=/path/to/datasets
root_data_dir = os.environ.get('ROOT_DATA_DIR', 'data')

lang_model_name = 'openai/clip-vit-base-patch32'

image_size = (640, 640)

weak_ratio_range = (0.5, 1.5)

mosaic_ratio_range = (1.0, 3.0)

train_multi_scales = None

coco_od_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2))

crowdhuman_metainfo = dict(classes=('person',), palette=[(220, 20, 60)])
crowdhuman_train_dataset = dict(
    type='CocoDataset',
    data_root=f'{root_data_dir}/crowdhuman/',
    data_prefix=dict(img='Images/'),
    ann_file='annotations/crowdhuman_train.json',
    metainfo=crowdhuman_metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=coco_od_train_pipeline,
    return_classes=True)

hiertext_metainfo = dict(classes=('text',), palette=[(255, 0, 0)])
hiertext_train_dataset = dict(
    type='CocoDataset',
    data_root=f'{root_data_dir}/hiertext/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    metainfo=hiertext_metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=coco_od_train_pipeline,
    return_classes=True)

objv1_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(backend_args=None),
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=f'{root_data_dir}/objects365v1/o365v1_label_map.json',
    max_tokens=256)

o365v1_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/objects365v1/',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=objv1_train_pipeline,
    return_classes=True,
    backend_args=None)

oi_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(backend_args=None),
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=f'{root_data_dir}/open_image/annotations/openimages_label_map.json',
    max_tokens=256)
oiv6_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/open_image/',
    data_prefix=dict(img='images/'),
    ann_file='annotations/oidv6-train-annotations_od.json',
    label_map_file='annotations/openimages_label_map.json',
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=oi_train_pipeline,
    return_classes=True,
    backend_args=None)

v3d_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(backend_args=None),
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=f'{root_data_dir}/v3det/annotations/v3det_2023_v1_label_map.json',
    max_tokens=256)
v3det_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/v3det/',
    data_prefix=dict(img=''),
    ann_file='annotations/v3det_2023_v1_train_od.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=v3d_train_pipeline,
    return_classes=True,
    backend_args=None)

bamboo1M_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(imdecode_backend='cv2', ignore_empty=True),
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=f'{root_data_dir}/bamboo-1M/annotations/bamboo_label_map.json',
    max_tokens=256)
bamboo1M_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/bamboo-1M/',
    data_prefix=dict(img='images/'),
    ann_file='annotations/bamboo_od.json',
    label_map_file='annotations/bamboo_label_map.json',
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=bamboo1M_train_pipeline,
    return_classes=True,
    backend_args=None)

sa_1b_3m_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    mosaic_ratio_range=mosaic_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(imdecode_backend='cv2', ignore_empty=True),
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=f'{root_data_dir}/sa-1b/annotations/sa-1b_3m_label_map.json',
    max_tokens=256)
sa_1b_3m_sampled_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/sa-1b/',
    data_prefix=dict(img='images/'),
    ann_file='annotations/sa-1b_3m_od_sampled_1500000.json',
    label_map_file='annotations/sa-1b_3m_label_map.json',
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=sa_1b_3m_train_pipeline,
    return_classes=True,
    backend_args=None)

vg_train_pipeline = build_fixed_scale_od_pipeline(
    image_size=image_size,
    weak_ratio_range=weak_ratio_range,
    multi_scales=train_multi_scales,
    load_image_kwargs=dict(imdecode_backend='cv2', ignore_empty=True),
    with_mosaic=False,
    with_random_flip=False,  # keep existing behavior
    with_filter=True,
    filter_min_gt_bbox_wh=(1e-2, 1e-2),
    with_hsv_random_aug=False,
    with_text_sampling=True,
    tokenizer_name=lang_model_name,
    num_sample_negative=85,
    label_map_file=None,
    max_tokens=256)

flickr30k_dataset_30k = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/flickr30k_entities/',
    data_prefix=dict(img='images/'),
    ann_file='flickr_30k_vg.json',
    label_map_file=None,
    filter_cfg=dict(),
    pipeline=vg_train_pipeline,
    return_classes=True,
    backend_args=None)

gqa_dataset_46k = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/gqa/',
    data_prefix=dict(img='images/'),
    ann_file='gqa_46k_vg.json',
    label_map_file=None,
    filter_cfg=dict(),
    pipeline=vg_train_pipeline,
    return_classes=True,
    backend_args=None)

BambooCLS_3m_sampled_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/bamboo-cls/',
    data_prefix=dict(img='images-3M/'),
    ann_file='annotations/bamboo-cls_3m_vg_sampled_1500000.json',
    label_map_file=None,
    filter_cfg=dict(),
    pipeline=vg_train_pipeline,
    return_classes=True,
    backend_args=None)
cc3m_dataset = dict(
    type='ODVGDataset',
    data_root=f'{root_data_dir}/cc3m/',
    data_prefix=dict(img='train/'),
    ann_file='cc3m_vg.json',
    label_map_file=None,
    filter_cfg=dict(),
    pipeline=vg_train_pipeline,
    return_classes=True,
    backend_args=None)

coco_od_test_pipeline = build_test_pipeline(
    scale=image_size, with_pad=True, pad_size=image_size)

coco_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    backend_args=None)

coco2017_val_dataset = dict(
    type='CocoDataset',
    data_root=f'{root_data_dir}/coco/',
    data_prefix=dict(img='val2017/'),
    ann_file='annotations/instances_val2017.json',
    test_mode=True,
    pipeline=coco_od_test_pipeline,
    return_classes=True,
    backend_args=None)

coco2017_val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{root_data_dir}/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
