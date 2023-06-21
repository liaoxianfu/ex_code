from mmcv.runner import CheckpointLoader
from collections import OrderedDict

import torch
import argparse
import mmcv
import os.path as osp


def convert_mobilevit(checkpoint):
    new_ckpt = OrderedDict()
    for k, v in checkpoint.items():
        new_k = k
        if new_k.startswith('stages.'):
            new_k = new_k.replace('stages.', 'stages_')
        new_k = 'model.' + new_k
        new_ckpt[new_k] = v
        print("convert key: {} --> {}".format(k, new_k))
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(description='Convert keys in timm pretrained mobilevit/mobilevutv2 models to'
                                     'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    
    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_mobilevit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
