_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth' 
checkpoint_file = 'pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth' 
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=7),
    auxiliary_head=dict(in_channels=512, num_classes=7))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00001,
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