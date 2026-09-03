# O365 + GoldG (Flickr30k×2 + GQA×2) text/visual pretrain subset.
_base_ = 'opus_dinov3_convnext-b_pretrain_o365.py'

model = dict(
    bbox_head=dict(
        vpg_loss_cfg=dict(
            _delete_=True,
            vl_align=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0,
            ),
            dec_ica=dict(
                type='VPGContentContrastiveLoss',
                temperature=0.07,
                loss_weight=1.0,
            ),
        ),
    ),
)

flickr30k_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=_base_.flickr30k_dataset_30k)
gqa_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=_base_.gqa_dataset_46k)

text_datasets_w_size = [
    [_base_.o365v1_dataset, -1],
    [flickr30k_dataset, -1],
    [gqa_dataset, -1],
]

visual_datasets_w_size = [
    [_base_.o365v1_dataset, -1],
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=[d[1] for d in text_datasets_w_size]),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[d[0] for d in text_datasets_w_size]))

alt_train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=[d[1] for d in visual_datasets_w_size]),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[d[0] for d in visual_datasets_w_size]))

# max_iter = 120000
max_iter = 100000
base_lr = 4e-4
val_interval = 10000
by_epoch = False
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=by_epoch, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=by_epoch,
        milestones=[int(0.5 * max_iter), int(0.9 * max_iter)],
        gamma=0.1)
]

train_cfg = dict(
    _delete_=True,
    type='OPUSIterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=val_interval,
    alt_interval=8,
    alt_mode=['text_only', 'text_visual'],
    dataloader_alt=alt_train_dataloader)

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

auto_scale_lr = dict(enable=False, base_batch_size=128)

default_hooks = dict(
    checkpoint=dict(by_epoch=by_epoch, interval=val_interval, max_keep_ckpts=10),
    visualization=dict(type='GroundingVisualizationHook'))
