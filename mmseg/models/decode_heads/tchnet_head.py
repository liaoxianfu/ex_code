# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .aspp_head import ASPPModule

@HEADS.register_module()
class TchHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6),dilations=(1, 6, 12, 18), **kwargs):
        super(TchHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        #############增加ASPPmodule 开始###################    
        self.aspp_module = ASPPModule(
            dilations,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg          
            )
        self.aspp_dwon_chs = nn.Conv2d(self.channels*4,self.channels,kernel_size=1,stride=1)
        self.tch_dwon_chs = nn.Conv2d(self.channels*2,self.channels,kernel_size=1,stride=1)
        #############增加ASPPmodule 结束###################    
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.conv_down_chs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            conv_dwon_chs = nn.Conv2d(self.channels*2,self.channels,kernel_size=1,stride=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            #########添加1x1 卷积 降低通道数量#########
            self.conv_down_chs.append(conv_dwon_chs)


        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output
    
    ##########ASPP 开始#############
    def aspp_forward(self,inputs):
        x = inputs[-1]
        aspp_list = self.aspp_module(x)
        out = torch.cat(aspp_list,dim=1)
        out = self.aspp_dwon_chs(out)
        return out
    ##########ASPP 结束#############


    def tch_forword(self,inputs):
        psp_out = self.psp_forward(inputs)
        aspp_out = self.aspp_forward(inputs)
        out = torch.cat((psp_out,aspp_out),dim=1)
        out = self.tch_dwon_chs(out)
        return out

    

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
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.tch_forword(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]

            ##################进行修改开始########################
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            upsample_la = resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            tmp  = torch.cat((laterals[i-1],upsample_la),dim=1)
            laterals[i-1] = self.conv_down_chs[i-1](tmp)
            ##################进行修改结束########################
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        # fpn_outs.append(aspp_out)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
