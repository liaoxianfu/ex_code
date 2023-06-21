#--------------------------------------------------------
#根据自己的数据特性，自写数据集转化, 标签图片必须是png
#-------------------------------------------------------
import os
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


# 对应关系


# ++++++++++++++++++++++++++++++++isaid++++++++++++++++++++++++++++++++++++#
# 裁剪不足的用255填充了
ISAID_CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')

ISAID_PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]
isaid_dict = {k:v for k,v in enumerate(ISAID_PALETTE)}
isaid_dict[255] = [255,255,255]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


# ++++++++++++++++++++++++++++++++loveda++++++++++++++++++++++++++++++++++++#

loveDA_CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
               'agricultural')

loveDA_PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]
loveda_dict = {k+1:v for k,v in enumerate(loveDA_PALETTE)}
loveda_dict[0] = [0,0,0]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#



# ++++++++++++++++++++++++++++++++potsdam++++++++++++++++++++++++++++++++++++#
potsdam_CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

potsdam_PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

potsdam_dict = {k+1:v for k,v in enumerate(potsdam_PALETTE)}
potsdam_dict[0] = [0,0,0]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


label2color_dict = {
    "isaid": isaid_dict,
    "loveda":loveda_dict,
    "potsdam":potsdam_dict
}

def convert_img(src,target,dataset):
    assert src is not None and src != ""
    assert target is not None and target != ""
    assert dataset is not None and dataset != ""
    assert dataset in label2color_dict

    palette_dict = label2color_dict[dataset]
    if not osp.exists(target):
        os.makedirs(target)
    png_names = os.listdir(src)
    print("正在遍历全部标签。")
    for png_name in tqdm(png_names):
        png = Image.open(os.path.join(src, png_name)) # RGB
        w, h = png.size
        # ----------------(gray--->gray)---------------------
        png = np.array(png, np.uint8)  # 输入为灰度 h, w
        out_png = np.zeros([h, w, 3])  # 新建的RGB为输出 h, w, c
        # 关系映射
        for i in range(png.shape[0]):  # i for h
            for j in range(png.shape[1]):
                color = palette_dict[png[i, j]]
                out_png[i, j, 0] = color[0]
                out_png[i, j, 1] = color[1]
                out_png[i, j, 2] = color[2]

        # print("out_png:", out_png.shape)
        out_png = Image.fromarray(np.array(out_png, np.uint8)) # 再次转化为Imag进行保存
        out_png.save(os.path.join(target, png_name))


def parse_args():
    parser = argparse.ArgumentParser(description='cover ann_file to palette img')
    parser.add_argument('--src',type=str,help="ann img dir")
    parser.add_argument('--target',type=str,help="palette img saved dir")
    parser.add_argument('--dataset',type=str,help="dataset name eg. loveda, potsdam, isaid")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    src = args.src
    target = args.target
    dataset = args.dataset
    print("src={}, target={}, dataset={}".format(src,target,dataset))
    convert_img(src,target,dataset)
    

