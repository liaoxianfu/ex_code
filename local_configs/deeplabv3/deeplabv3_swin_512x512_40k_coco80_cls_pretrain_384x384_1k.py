_base_ = [
    '../_base_/models/deeplabv3_swin.py',
    '../_base_/datasets/coco80_cls.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=1024, num_classes=80),
    auxiliary_head=dict(in_channels=512, num_classes=80))