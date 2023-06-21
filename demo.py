from timm.models import mobilevit
import timm
import cv2
import torch
import numpy as np
from PIL import Image
from mmseg.datasets import PotsdamDataset
from mmcv.runner import CheckpointLoader
from collections import OrderedDict
import torch
import argparse

_cfg = dict

default_cfgs = {
    'mobilevit_xxs': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_xxs-ad385b40.pth'),
    'mobilevit_xs': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_xs-8fbd6366.pth'),
    'mobilevit_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_s-38a5a959.pth'),
    'semobilevit_s': _cfg(),

    'mobilevitv2_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_050-49951ee2.pth',
        crop_pct=0.888),
    'mobilevitv2_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_075-b5556ef6.pth',
        crop_pct=0.888),
    'mobilevitv2_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_100-e464ef3b.pth',
        crop_pct=0.888),
    'mobilevitv2_125': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_125-0ae35027.pth',
        crop_pct=0.888),
    'mobilevitv2_150': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150-737c5019.pth',
        crop_pct=0.888),
    'mobilevitv2_175': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175-16462ee2.pth',
        crop_pct=0.888),
    'mobilevitv2_200': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200-b3422f67.pth',
        crop_pct=0.888),

    'mobilevitv2_150_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150_in22ft1k-0b555d7b.pth',
        crop_pct=0.888),
    'mobilevitv2_175_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175_in22ft1k-4117fa1f.pth',
        crop_pct=0.888),
    'mobilevitv2_200_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200_in22ft1k-1d7c8927.pth',
        crop_pct=0.888),

    'mobilevitv2_150_384_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150_384_in22ft1k-9e142854.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'mobilevitv2_175_384_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175_384_in22ft1k-059cbe56.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'mobilevitv2_200_384_in22ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200_384_in22ft1k-32c87503.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
}



data = torch.rand((2,3,512,512))
m = timm.create_model('mobilevit_xs', features_only=True,out_indices=(0,1,2,3,4))
a = m(data)
for x in a:
    print(x.shape)


# encoder_channels =m.feature_info.channels()
# print(encoder_channels)

# model_weight = 'mobilevit_s.pth'
# pretrained_dict=torch.load(model_weight)
# model_dict=m.state_dict()
# # 1. filter out unnecessary keys
# new_dict = {}
# for k,v in pretrained_dict.items():
#     if k in model_dict:
#         new_dict[k] = v
#         print("init ",k)
#     else:
#         print(k,"not in model_dict")

# model_dict.update(new_dict)
# m.load_state_dict(model_dict)

# # m.load_state_dict(torch.load('mobilevit_s.pth'),strict=False
# # )


# # # m = mobilevit.mobilevitv2_125(pretrained=True,out_indices=(0,1,2,3,4,5))
# # for param_tensor in m.state_dict():
# #     print(param_tensor, "\t", m.state_dict()[param_tensor].size())
# a = m(data)
# for x in a:
#     print(x.shape)
# checkpoint = CheckpointLoader.load_checkpoint('pretrain/mobilevit_s-38a5a959.pth', map_location='cpu')

# new_ckpt = OrderedDict()
# for k,v in checkpoint.items():
#     new_k = k
#     if new_k.startswith('stages.'):
#         new_k = new_k.replace('stages.','stages_')
    
#     new_k = 'model.'+new_k
#     new_ckpt[new_k] = v

# torch.save(new_ckpt,'mobilevit_s.pth')


