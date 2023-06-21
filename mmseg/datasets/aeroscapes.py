# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AeroscapesDataset(CustomDataset):
    """Aeroscapes Dataset

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Background', 'Person', 'Bike', 'Car', 'Drone', 'Boat',
               'Animal', 'Obstacle', 'Construction', 'Vegetation', 'Road',
               'Sky')

    PALETTE = [[0, 0, 0], [192, 128, 128], [0, 128, 0], [128, 128, 128],
               [128, 0, 0], [0, 0, 128], [192, 0, 128], [192, 0, 0],
               [192, 128, 0], [0, 64, 0], [128, 128, 0], [0, 128, 128]]

    def __init__(self, split, **kwargs):
        super(AeroscapesDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            split=split,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
