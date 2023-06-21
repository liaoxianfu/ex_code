_base_ = [
    '../_base_/models/lightdw_mobilevit.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint_file = 'pretrain/mobilevit/mobilevit_xxs.pth'
model = dict(
    backbone = dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        model_name='mobilevit_xxs',
        out_indices=(0,1,2,3,4)
    ),
    decode_head=dict(in_channels=[24, 48, 64, 320], num_classes=6))



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
data = dict(samples_per_gpu=4)
evaluation = dict(interval=16000, metric=['mIoU','mFscore'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=16000)
