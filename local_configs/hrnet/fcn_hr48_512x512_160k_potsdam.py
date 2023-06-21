_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=6))

model = dict(pretrained='open-mmlab://msra/hrnetv2_w48',
             backbone=dict(extra=dict(stage2=dict(num_channels=(48, 96)),
                                      stage3=dict(num_channels=(48, 96, 192)),
                                      stage4=dict(num_channels=(48, 96, 192, 384)))),
             decode_head=dict(in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

data = dict(samples_per_gpu=2)
evaluation = dict(interval=100, metric=['mIoU','mFscore'], pre_eval=True)