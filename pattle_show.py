#--------------------------------------------------------
#根据自己的数据特性，自写数据集转化, 标签图片必须是png
#-------------------------------------------------------
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

Origin_SegmentationClass_path = "/home/b104/liao/data/zip_file/Potsdam/zip/potsdam512/out/ann_dir/val"
Out_SegmentationClass_path = "/home/b104/liao/data/zip_file/Potsdam/zip/potsdam512/out/ann_dir/val_out"

# 对应关系
# Origin_Point_Value = np.array([0, 1, 2, 3])
# Out_Point_Value = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]])

PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
# PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
# label2color_dict = {
#     0: [0, 0, 0],
#     1: [255, 255, 255],
#     2: [255, 0, 0],
#     3: [255, 255, 0],
#     4: [0, 0, 255],
#     5: [159, 129, 183],
#     6: [0, 255, 0],
#     7: [255, 195, 128]
# }

label2color_dict = {
    #0: [0, 0, 0],
    0: [255, 255, 255],
    1: [0, 0, 255],
    2: [0, 255, 255],
    3: [0, 255, 0],
    4: [255, 255, 0],
    5: [255, 0, 0]
}

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)
    #
    png_names = os.listdir(Origin_SegmentationClass_path)  # 获得图片的文件名
    print("正在遍历全部标签。")
    for png_name in tqdm(png_names):
        png = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))  # RGB
        w, h = png.size
        # ----------------(gray--->gray)---------------------
        png = np.array(png, np.uint8)  # 输入为灰度 h, w
        out_png = np.zeros([h, w, 3])  # 新建的RGB为输出 h, w, c
        # 关系映射
        for i in range(png.shape[0]):  # i for h
            for j in range(png.shape[1]):
                color = label2color_dict[png[i, j]]
                out_png[i, j, 0] = color[0]
                out_png[i, j, 1] = color[1]
                out_png[i, j, 2] = color[2]

        # print("out_png:", out_png.shape)
        out_png = Image.fromarray(np.array(out_png, np.uint8))  # 再次转化为Imag进行保存
        out_png.save(os.path.join(Out_SegmentationClass_path, png_name))
'''
In order to further enhance the ability of model segmentation, we replaced the decoder from UperNetDecoder to our designed MFFDecoder and conducted experiments on iSAID dataset, as shown in Table 1.


'''
