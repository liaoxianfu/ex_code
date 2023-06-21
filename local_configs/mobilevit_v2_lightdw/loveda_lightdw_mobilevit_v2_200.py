_base_ = [
    '../_base_/models/lightdw_mobilevit_v2.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint_file = 'pretrain/mobilevit/mobilevitv2_200_384_in22ft1k.pth'
model = dict(
    backbone = dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        model_name='mobilevitv2_200_384_in22ft1k',
        out_indices=(0,1,2,3,4)
    ),
    decode_head=dict(
           loss_decode=[dict(type='FocalLoss', loss_name='loss_focal', loss_weight=4.0),
                        dict(type='CrossEntropyLoss', loss_name='loss_ce',loss_weight=1.0)],
                   in_index=(2,3,4),
                   in_channels=[512,768,1024], num_classes=7,
        )
    )



optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=1e-6,
    by_epoch=False)


data = dict(samples_per_gpu=4)
evaluation = dict(interval=16000, metric=['mIoU','mFscore'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=16000)


"""

与全局场景上下文提取模块的思想类似，轻量级全局场景上下文Light GSCI Module的目标也是通过不同尺度的卷积捕获不同分辨率的特征信息作为多尺度上下文，通过上下文将一个位置上的像素与该位置不同尺度范围内的像素进行建模，提升特征图中的语义信息
与全局场景上下文提取模块的思想类似，轻量级全局场景上下文Light GSCI Module的目标也是通过不同尺度的卷积捕获不同分辨率的特征信息作为多尺度上下文，通过上下文将一个位置上的像素与该位置不同尺度范围内的像素进行建模，提升特征图中的语义信息
与全局场景上下文提取模块的思想类似，轻量级全局场景上下文LightGSCIModule的目标也是通过不同尺度的卷积捕获不同分辨率的特征信息作为多尺度上下文，通过上下文将一个位置上的像素与该位置不同尺度范围内的像素进行建模，以提升特征图中的语义信息。
"""