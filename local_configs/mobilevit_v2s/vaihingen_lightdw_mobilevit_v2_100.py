_base_ = [
    '../_base_/models/lightdw_mobilevit_v2s.py', '../_base_/datasets/loveda1024.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

checkpoint_file = 'pretrain/mobilevit/mobilevitv2_100.pth'
model = dict(backbone=dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    model_name='mobilevitv2_100',
),
decode_head=dict(in_channels=[128, 256, 384, 512], num_classes=6),
auxiliary_head=dict(in_channels=384, num_classes=6))

optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=1e-6,
                 by_epoch=False)

data = dict(samples_per_gpu=8)
evaluation = dict(interval=16000, metric=['mIoU', 'mFscore'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=16000)
