import argparse
import os
import shutil

import imgviz
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

"""'
coco数据集语义分割工具,将数据集转化为VOC格式的数据:
coco数据集本身没有提供语义分割后的数据集,只有json文件表示语义分割的数据
这样训练非常的不方便,通过该工具可以生成数据集
数据目录如下所示
├── annotations
│  ├── captions_train2017.json
│  ├── captions_val2017.json
│  ├── instances_train2017.json
│  ├── instances_val2017.json
│  ├── person_keypoints_train2017.json
│  ├── person_keypoints_val2017.json
│ 
├── images
│  ├── test2017
│  ├── train2017
│  └── val2017
|__________

转换的方法

python coco_seg_tools.py --coco_root=你的coco数据集的根路径(可以看到annotations目录)
--coco_seg_type={voc20_cls,coco80_cls,coco90_cls}

其中 voc20_cls是指定标注为voc 相同的20类,其他的不进行mask 如果一张图中没有这里的20类就不会保存到该数据集中
coco80_cls 是保存coco80类 也就是去除不存在的类 像素索引为 0-80
coco90_cls 是保存90类 像素索引为0-90

"""


# fmt:off
# 原始的coco 90类 缺失不处理
coco90_cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

# 处理成为coco80 
coco80_cls = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

# 提取和voc20类相同的数据
voc20_cls = [0,5,2,16,9,44,6,3,17,62,21,67,18,19,4,1,64,20,63,7,72]

coco_dir_info = '''
├── annotations
│  ├── captions_train2017.json
│  ├── captions_val2017.json
│  ├── instances_train2017.json
│  ├── instances_val2017.json
│  ├── person_keypoints_train2017.json
│  ├── person_keypoints_val2017.json
│ 
├── images
│  ├── test2017
│  ├── train2017
│  └── val2017
'''


# fmt:on


class COCOSegUtil:
    def __init__(
        self, coco_root, coco_seg_type="voc20_cls", data_type="train2017"
    ) -> None:
        self.class_set = set()
        self.process_img_count = 0
        self.coco_root = coco_root
        self.choose_seg_type(coco_seg_type)
        assert (
            data_type == "train2017" or data_type == "val2017"
        ), "only support coco2017"
        self.data_type = data_type
        annotation_file = os.path.join(
            coco_root, "annotations", "instances_{}.json".format(data_type)
        )
        self.coco_seg_type = coco_seg_type
        self.coco = COCO(annotation_file)
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        print("catIds len:{}, imgIds len:{}".format(len(self.catIds), len(self.imgIds)))

    def choose_seg_type(self, coco_seg_type):
        if coco_seg_type == "voc20_cls":
            self.seg_cls = voc20_cls
        elif coco_seg_type == "coco80_cls":
            self.seg_cls = coco80_cls
        elif coco_seg_type == "coco90_cls":
            self.seg_cls = coco90_cls
        else:
            raise Exception("type is voc20_cls, coco80_cls or coco90_cls")

    def process(self):
        os.makedirs(
            os.path.join(
                self.coco_root, self.coco_seg_type, self.data_type, "SegmentationClass"
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                self.coco_root, self.coco_seg_type, self.data_type, "JPEGImages"
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                self.coco_root, self.coco_seg_type, "ImageSets", "Segmentation"
            ),
            exist_ok=True,
        )
        for imgId in tqdm(self.imgIds):
            img_list = self.coco.loadImgs(imgId)
            for img in img_list:
                status, mask = self.get_mask(img)
                if status:
                    img_origin_path = os.path.join(
                        self.coco_root, "images", self.data_type, img["file_name"]
                    )
                    img_output_path = os.path.join(
                        self.coco_root,
                        self.coco_seg_type,
                        self.data_type,
                        "JPEGImages",
                        img["file_name"],
                    )
                    seg_output_path = os.path.join(
                        self.coco_root,
                        self.coco_seg_type,
                        self.data_type,
                        "SegmentationClass",
                        img["file_name"].replace(".jpg", ".png"),
                    )
                    shutil.copy(img_origin_path, img_output_path)
                    self.class_set.add(mask.max())
                    self.save_colored_mask(mask, seg_output_path)
                    self.process_img_count += 1
        self.get_img_name()  # 将图片名称保存到文件中
        print("类别集合为{}".format(self.class_set))
        print("一共生成了{}张图".format(self.process_img_count))

    def get_img_name(self):
        """
        获取处理好的文件名称
        """
        lable_name = self.data_type.replace("2017", "")
        img_name_text = os.path.join(
            self.coco_root,
            self.coco_seg_type,
            "ImageSets",
            "Segmentation",
            "{}.txt".format(lable_name),
        )
        img_dir = os.path.join(
            self.coco_root, self.coco_seg_type, self.data_type, "JPEGImages",
        )
        with open(img_name_text, "a") as f:
            for img_name in os.listdir(img_dir):
                img_id = img_name.replace(".jpg", "\n")
                f.write(img_id)

    def get_mask(self, img):
        annIds = self.coco.getAnnIds(imgIds=img["id"], catIds=self.catIds, iscrowd=None)
        if len(annIds) <= 0:
            return False, None
        anns = self.coco.loadAnns(annIds)
        anns_voc = []
        for an in anns:
            if an["category_id"] in self.seg_cls:
                anns_voc.append(an)
        if len(anns_voc) <= 0:
            return False, None
        mask = np.zeros((img["height"], img["width"]))
        for an in anns_voc:
            category_id = self.seg_cls.index(an["category_id"])
            mask = np.maximum(mask, self.coco.annToMask(an) * category_id)
        return True, mask

    @staticmethod
    def save_colored_mask(mask, save_path):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(save_path)


class COCOSegTool:
    def __init__(self, coco_root, coco_seg_type="voc20_cls") -> None:
        self.coco_root = coco_root
        # 判断是否按照规定的文件夹存放
        img_dir = os.path.join(coco_root, "images")
        if not os.path.exists(img_dir):
            raise Exception(
                f"please make sure exists dir {img_dir} \n img_dir like this \n {coco_dir_info}"
            )
        self.coco_seg_type = coco_seg_type
        seg_type_root = os.path.join(coco_root, coco_seg_type)
        assert not os.path.exists(
            seg_type_root
        ), f"{seg_type_root} has exists,please remove it"
        self.train_data_type = "train2017"
        self.val_data_type = "val2017"
        self.coco_train = COCOSegUtil(
            coco_root, coco_seg_type, data_type=self.train_data_type
        )
        self.coco_val = COCOSegUtil(
            coco_root, coco_seg_type, data_type=self.val_data_type
        )

    def process(self):
        self.coco_val.process()
        print("验证数据集处理完成")
        self.coco_train.process()
        print("测试数据集处理完成")
        self.merge()
        print("合并完成")

    def merge(self):
        target_jpeg_path = os.path.join(
            self.coco_root, self.coco_seg_type, "JPEGImages"
        )
        os.makedirs(target_jpeg_path, exist_ok=True)
        target_seg_path = os.path.join(
            self.coco_root, self.coco_seg_type, "SegmentationClass"
        )
        os.makedirs(target_seg_path, exist_ok=True)
        train_jpeg_src = os.path.join(
            self.coco_root, self.coco_seg_type, self.train_data_type, "JPEGImages"
        )
        train_seg_src = os.path.join(
            self.coco_root,
            self.coco_seg_type,
            self.train_data_type,
            "SegmentationClass",
        )

        jpeg_imgs = os.listdir(train_jpeg_src)
        for img in tqdm(jpeg_imgs):
            src_img = os.path.join(train_jpeg_src, img)
            target_img = os.path.join(target_jpeg_path, img)
            shutil.move(src_img, target_img)

        seg_imgs = os.listdir(train_seg_src)
        for img in tqdm(seg_imgs):
            src_img = os.path.join(train_seg_src, img)
            target_img = os.path.join(target_seg_path, img)
            shutil.move(src_img, target_img)

        val_jpeg_src = os.path.join(
            self.coco_root, self.coco_seg_type, self.val_data_type, "JPEGImages"
        )
        jpeg_imgs = os.listdir(val_jpeg_src)
        for img in tqdm(jpeg_imgs):
            src_img = os.path.join(val_jpeg_src, img)
            target_img = os.path.join(target_jpeg_path, img)
            shutil.move(src_img, target_img)
        val_seg_src = os.path.join(
            self.coco_root, self.coco_seg_type, self.val_data_type, "SegmentationClass"
        )
        seg_imgs = os.listdir(val_seg_src)
        for img in tqdm(seg_imgs):
            src_img = os.path.join(val_seg_src, img)
            target_img = os.path.join(target_seg_path, img)
            shutil.move(src_img, target_img)
        # 删除文件夹
        shutil.rmtree(os.path.join(self.coco_root, self.coco_seg_type, "train2017"))
        shutil.rmtree(os.path.join(self.coco_root, self.coco_seg_type, "val2017"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        default="../../data/coco",
        type=str,
        help="coco dataset directory",
    )
    parser.add_argument(
        "--coco_seg_type",
        default="voc20_cls",
        type=str,
        help="type is voc20_cls, coco80_cls or coco90_cls",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    coco_seg_tool = COCOSegTool(args.coco_root, coco_seg_type=args.coco_seg_type)
    coco_seg_tool.process()

