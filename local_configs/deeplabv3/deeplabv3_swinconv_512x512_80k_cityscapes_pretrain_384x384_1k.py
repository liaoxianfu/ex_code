_base_ = [
    '../_base_/models/deeplabv3_swinconv.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

checkpoint_file = 'pretrain/swin_base_patch4_window12_384_20220317-55b0104a.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        conv_attn=(0, 2, 5, 3),
        window_size=12),
    decode_head=dict(in_channels=1024, num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
evaluation = dict(interval=1600, metric='mIoU', pre_eval=True)

# mixed precision
fp16 = dict(loss_scale='dynamic')
data = dict(samples_per_gpu=2)