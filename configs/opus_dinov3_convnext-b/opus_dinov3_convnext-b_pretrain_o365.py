_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../datasets/640_640.py'
]

custom_imports = dict(imports=['opus'], allow_failed_imports=False)

num_levels = 3
in_channels = [256, 512, 1024]

model = dict(
    type='OPUS',
    num_feature_levels=num_levels,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    use_cache_label=True,
    use_visual_pre_encoder=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='CLIPModel',
        name='openai/clip-vit-base-patch32',
        max_tokens=77,
        pad_to_max=False,
        with_projection=False,
        frozen_stages=-1,
    ),
    backbone=dict(
        type='HFDINOv3ConvNeXtBackbone',
        model_name='facebook/dinov3-convnext-base-pretrain-lvd1689m',
        out_indices=(2, 3, 4), # 1/8 1/16 1/32
        train_backbone=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=in_channels,
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels,
    ),
    visual_prompt_model=dict(
        num_layers=3,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        post_norm_cfg=None,
    ),
    encoder=dict(
        type='OPUSHybridEncoder',
        rt_detr_hybrid_cfg=dict(
            attn_encode_idx=[2],
            num_feature_levels=num_levels,
            fusion_expansion=1,
            fusion_depth=3,
        ),
        num_layers=1,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=1, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='OPUSHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        vpg_loss_cfg=dict(
            vl_align=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0,
            ),
        ),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(
        chunked_size=-1,
        max_per_img=300,
        inference_head_last_layer_only=True))

datasets = [_base_.o365v1_dataset]
dataset_size = [-1] * len(datasets)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=dataset_size),
    dataset=dict(type='ConcatDataset', datasets=datasets))

val_dataloader = dict(dataset=_base_.coco2017_val_dataset)
test_dataloader = val_dataloader
val_evaluator = _base_.coco2017_val_evaluator
test_evaluator = val_evaluator

base_lr = 4e-4
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

by_epoch = True
if by_epoch:
    max_epochs = 30
    val_interval = 1
    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
    param_scheduler = [
        dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
        dict(
            type='MultiStepLR',
            begin=0,
            end=max_epochs,
            by_epoch=by_epoch,
            milestones=[int(0.64 * max_epochs), int(0.87 * max_epochs)],
            gamma=0.1)
    ]
else:
    max_iter = 400000
    val_interval = 10000
    train_cfg = dict(
        _delete_=True,
        type='OPUSIterBasedTrainLoop',
        max_iters=max_iter,
        val_interval=val_interval,
    )
    param_scheduler = [
        dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
        dict(
            type='MultiStepLR',
            begin=0,
            end=max_iter,
            by_epoch=by_epoch,
            milestones=[int(0.64 * max_iter), int(0.87 * max_iter)],
            gamma=0.1)
    ]

auto_scale_lr = dict(enable=False, base_batch_size=128)

default_hooks = dict(
    checkpoint=dict(by_epoch=by_epoch, interval=val_interval, max_keep_ckpts=10),
    visualization=dict(type='GroundingVisualizationHook'),
)

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl',
        timeout=3600,
    ),
)
