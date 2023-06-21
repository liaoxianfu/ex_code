import torch.nn as nn
from mmcv.runner import BaseModule


class ConvAttention3(BaseModule):
    """
    创建卷积注意力机制
    """

    def __init__(self, conv_num, in_channel, out_channel, topper=False, init_cfg=None):
        self.conv_num = conv_num
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.init_cfg = init_cfg
        super().__init__(init_cfg)
        self.layers = []
        if topper:
            self.layers += [
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
            ]
        else:
            self.layers += [
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(),
            ]
        for _ in range(1, self.conv_num):
            self.layers += [
                nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(),
            ]
        self.layers += [
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)  # 向上取整
        ]
        self.conv_attention = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.conv_attention(x)
        out_size = (x.shape[2], x.shape[3])
        # x = x.flatten(2).transpose(1, 2)
        return x, out_size