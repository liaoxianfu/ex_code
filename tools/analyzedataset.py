import os.path as osp
import os
from collections import Counter
import cv2

from tqdm import tqdm

path = 'data/potsdam/ann_dir/train/7_11_1536_4096_2048_4608.png'


def get_single_img(img_path):
    img = cv2.imread(img_path)
    data = img.flatten()
    res = Counter(data)
    return res


def process_data(dir):
    img_list = os.listdir(dir)
    data_dict = {}
    for i in tqdm(range(1, len(img_list) + 1)):
        res_dict = get_single_img(osp.join(dir, img_list[i - 1]))
        for res in res_dict.keys():
            if data_dict.get(res) == None:
                data_dict[res] = res_dict[res]
            else:
                data_dict[res] += res_dict[res]
    print("最终结果为 ", data_dict)


if __name__ == "__main__":
    process_data('data/vaihingen/ann_dir/train')
    #process_data('data/iSAID/ann_dir/train')
