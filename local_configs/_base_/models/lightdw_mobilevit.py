# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='Mobilevit',
                  model_name='mobilevit_s',
                  out_indices=(0,1,2,3,4)
                ),
    decode_head=dict(
        type='LightDWHead',
        in_channels=(64, 96, 128, 640),
        dilations=(1, 6, 12, 18),
        in_index=(1,2,3,4),
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# model = dict(
#     type='EncoderDecoder',
#     backbone=dict(
#         type='MobileNetV3',
#         arch='large',
#         out_indices=(1, 3, 16),
#         norm_cfg=norm_cfg),
#     decode_head=dict(
#         type='LRASPPHead',
#         in_channels=(16, 24, 960),
#         in_index=(0, 1, 2),
#         channels=128,
#         input_transform='multiple_select',
#         dropout_ratio=0.1,
#         num_classes=19,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='ReLU'),
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
