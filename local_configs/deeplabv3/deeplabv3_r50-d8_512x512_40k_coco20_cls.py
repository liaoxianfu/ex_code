_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/coco20_cls.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
