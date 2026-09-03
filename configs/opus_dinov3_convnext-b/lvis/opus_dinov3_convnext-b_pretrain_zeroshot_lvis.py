_base_ = '../opus_dinov3_convnext-b_pretrain_o365.py'
root_data_dir = _base_.root_data_dir

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl',
        timeout=7200,
    ),
)

model = dict(
    test_cfg=dict(
        max_per_img=300,
        chunked_size=40,
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
        ann_file='annotations/lvis_od_val.json',
        test_mode=True,
        pipeline=_base_.coco_od_test_pipeline,
        return_classes=True,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root + 'annotations/lvis_od_val.json',
    backend_args=None)
test_evaluator = val_evaluator
