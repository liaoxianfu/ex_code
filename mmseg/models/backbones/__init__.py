# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .swin_conv import SwinTransformerWithConv
from .swin_conv2 import SwinTransformerWithConv2
from .swin_conv3 import SwinTransformerWithConv3
from .swin_conv4 import SwinTransformerWithConv4
from .swin_conv5 import SwinTransformerWithConv5
from .swin_conv6 import SwinTransformerConv6
from .swin_conv7 import SwinTransformerConv7
from .swin_conv8 import SwinTransformerConv8
from .mobilevit import Mobilevit
from .mobilevit_v2 import MobilevitV2
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN', 'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet',
    'MobileNetV3', 'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer', 'BiSeNetV1', 'BiSeNetV2', 'ICNet',
    'TIMMBackbone', 'ERFNet', 'PCPVT', 'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'SwinTransformerWithConv',
    'SwinTransformerWithConv2', 'SwinTransformerWithConv3', 'SwinTransformerWithConv4', 'SwinTransformerWithConv5',
    'SwinTransformerConv6','SwinTransformerConv7','SwinTransformerConv8','Mobilevit','MobilevitV2'
]
