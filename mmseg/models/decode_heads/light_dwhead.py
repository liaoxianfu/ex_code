# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .aspp_head import ASPPModule


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU,
              self).__init__(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
                             nn.BatchNorm2d(out_channel), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


@HEADS.register_module()
class LightDWHead(BaseDecodeHead):
    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(LightDWHead, self).__init__(input_transform='multiple_select', **kwargs)

        #############增加ASPPmodule 开始###################
        self.aspp_module = ASPPModule(dilations,
                                      self.in_channels[-1],
                                      self.channels,
                                      conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)
        self.aspp_dwon_chs = nn.Conv2d(self.channels * len(dilations), self.channels, kernel_size=1, stride=1)
        self.tch_dwon_chs = ConvModule(self.channels * 2,
                                       self.channels,
                                       3,
                                       padding=1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg)
        #############增加ASPPmodule 结束###################
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels,
                                self.channels,
                                1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg,
                                inplace=False)
            fpn_conv = InvertedResidual(self.channels,self.channels,stride=1,expand_ratio=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            

        self.fpn_bottleneck = ConvModule(len(self.in_channels) * self.channels,
                                         self.channels,
                                         3,
                                         padding=1,
                                         conv_cfg=self.conv_cfg,
                                         norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg)

    ##########ASPP 开始#############
    def aspp_forward(self, inputs):
        x = inputs[-1]
        aspp_list = self.aspp_module(x)
        out = torch.cat(aspp_list, dim=1)
        out = self.aspp_dwon_chs(out)
        return out

    ##########ASPP 结束#############

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.aspp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]

            ##################进行修改开始########################
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            ##################进行修改结束########################
        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)

        # fpn_outs.append(aspp_out)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
