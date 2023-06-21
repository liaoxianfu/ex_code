from ..builder import BACKBONES
from mmcv.runner import BaseModule
import timm
import warnings


@BACKBONES.register_module()
class MobilevitV2(BaseModule):

    def __init__(self, model_name='mobilevitv2_200_384_in22ft1k', out_indices=(0, 1, 2, 3, 4), pretrained=None, init_cfg=None) -> None:
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super().__init__(init_cfg=init_cfg)

        model_list = [
            'mobilevitv2_050', 'mobilevitv2_075', 'mobilevitv2_100',
            'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_150_384_in22ft1k', 'mobilevitv2_150_in22ft1k', 'mobilevitv2_175',
            'mobilevitv2_175_384_in22ft1k', 'mobilevitv2_175_in22ft1k', 'mobilevitv2_200', 'mobilevitv2_200_384_in22ft1k',
            'mobilevitv2_200_in22ft1k'
        ]
        assert self.model_name in  model_list
        self.model_name = model_name
        self.out_indices = out_indices
        self.model = timm.create_model(model_name, features_only=True, pretrained=False, out_indices=out_indices)

    def forward(self, x):
        data_list = self.model(x)
        outs = []
        for i in range(len(data_list)):
            if i in self.out_indices:
                outs.append(data_list[i])
        return outs
