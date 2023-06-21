import argparse
import numpy as np
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg_img_path",
        default="/home/b104/liao/data/coco",
        type=str,
        help="img file",
    )
    return parser.parse_args()


def get_palette(seg_path):
    # 读取mask标签
    target = Image.open(seg_path)
    # 获取调色板
    palette = target.getpalette()
    palette = np.reshape(palette, (-1, 3)).tolist()
    print(palette)


if __name__ == "__main__":
    args = get_args()
    get_palette(args.seg_img_path)