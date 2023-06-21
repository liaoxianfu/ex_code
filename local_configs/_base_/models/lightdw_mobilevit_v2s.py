# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='Mobilevit', model_name='mobilevitv2_200_384_in22ft1k', out_indices=(0, 1, 2, 3, 4)),
    decode_head=dict(type='LightDWHead',
                     in_channels=(256, 512, 768, 1024),
                     dilations=(1, 8, 16, 32),
                     in_index=(1, 2, 3, 4),
                     channels=256,
                     dropout_ratio=0.1,
                     num_classes=19,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=[
                         dict(type='FocalLoss', loss_name='loss_focal', loss_weight=4.0),
                         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
                     ]),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=768,
                        in_index=3,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=19,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
