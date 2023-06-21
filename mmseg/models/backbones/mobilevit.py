from ..builder import BACKBONES
from mmcv.runner import BaseModule
import timm
import warnings


@BACKBONES.register_module()
class Mobilevit(BaseModule):
    def __init__(self,
                model_name='mobilevit_s',
                out_indices=(0,1,2,3,4),
                pretrained=None,
                init_cfg=None
                ) -> None:
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
        self.model_name =  model_name
        self.out_indices = out_indices
        self.model = timm.create_model(model_name, features_only=True,pretrained=False,
        out_indices=out_indices)
        
    def forward(self,x):
        data_list = self.model(x)
        outs = []
        for i in range(len(data_list)):
            if i in self.out_indices:
                outs.append(data_list[i])
        return outs

