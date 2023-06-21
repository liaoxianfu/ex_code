# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCO80Dataset(CustomDataset):
    """
    COCO data set convert like VOC dataset,remove not contains classes
    total classes is 80

     Args:
        split (str): Split txt file for COCO like Pascal VOC.
    """

    CLASSES = ('person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush')

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
               [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
               [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128],
               [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64],
               [128, 0, 64], [0, 128, 64], [128, 128, 64], [0, 0, 192],
               [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64],
               [192, 0, 64], [64, 128, 64], [192, 128, 64], [64, 0, 192],
               [192, 0, 192], [64, 128, 192], [192, 128, 192], [0, 64, 64],
               [128, 64, 64], [0, 192, 64], [128, 192, 64], [0, 64, 192],
               [128, 64, 192], [0, 192, 192], [128, 192, 192], [64, 64, 64],
               [192, 64, 64], [64, 192, 64], [192, 192, 64], [64, 64, 192],
               [192, 64, 192], [64, 192, 192], [192, 192, 192], [32, 0, 0],
               [160, 0, 0], [32, 128, 0], [160, 128, 0], [32, 0, 128],
               [160, 0, 128], [32, 128, 128], [160, 128, 128], [96, 0, 0],
               [224, 0, 0], [96, 128, 0], [224, 128, 0], [96, 0, 128],
               [224, 0, 128], [96, 128, 128], [224, 128, 128], [32, 64, 0]]

    def __init__(self, split, **kwargs):
        super(COCO80Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True, # 忽略标签为0的类
            split=split,
            **kwargs)
        assert len(COCO80Dataset.CLASSES) == len(COCO80Dataset.PALETTE)
        assert osp.exists(self.img_dir) and self.split is not None