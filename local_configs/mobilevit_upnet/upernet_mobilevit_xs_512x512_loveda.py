_base_ = [
    '../_base_/models/upernet_mobilevit.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint_file = 'pretrain/mobilevit/mobilevit_xxs.pth'
model = dict(
    backbone = dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        model_name='mobilevit_xs',
        out_indices=(0,1,2,3,4)
    ),
    decode_head=dict(in_channels=[48, 64, 80, 384], num_classes=7),
    auxiliary_head=dict(in_channels=80, num_classes=7))



optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
data = dict(samples_per_gpu=2)
evaluation = dict(interval=16000, metric=['mIoU','mFscore'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=16000)
