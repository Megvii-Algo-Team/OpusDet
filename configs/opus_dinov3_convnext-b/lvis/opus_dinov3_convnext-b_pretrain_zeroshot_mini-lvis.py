_base_ = '../opus_dinov3_convnext-b_pretrain_o365.py'
root_data_dir = _base_.root_data_dir
import os

chunked_size = os.getenv('chunked_size', 40)
max_per_img = os.getenv('max_per_img', 300)

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl',
        timeout=7200,
    ),
)

model = dict(
    test_cfg=dict(
        max_per_img=max_per_img,
        chunked_size=chunked_size,
    ),
)

data_root = f'{root_data_dir}/coco/'

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='LVISV1Dataset',
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        test_mode=True,
        pipeline=_base_.coco_od_test_pipeline,
        return_classes=True,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root + 'annotations/lvis_v1_minival_inserted_image_name.json',
    backend_args=None)
test_evaluator = val_evaluator
