_base_ = [
    '../_base_/models/upernet_swinconv6.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# checkpoint_file = 'pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        conv_depths=(1, 1, 3, 1)
        ),
       decode_head=dict(
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                     in_channels=[96, 192, 384, 768], num_classes=16
        ),
       auxiliary_head=dict(
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce',loss_weight=1.0),
                     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                     in_channels=384, num_classes=16),
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
evaluation = dict(interval=16000, metric=['mIoU','mFscore'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=16000)
