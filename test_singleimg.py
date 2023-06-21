import os
import os.path as osp

import mmcv

from mmseg.apis import inference_segmentor, init_segmentor

path = 'ex_res/ch04/Potsdam'

config_file = 'loss_potsdam_swin_base_conv6_160k.py'

config_file = osp.join(path, config_file)
checkpoint_file = 'last.pth'
checkpoint_file = osp.join(path, checkpoint_file)
# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

test_img_dir = osp.join(path, 'org')
out_img_dir = osp.join(path, 'pre')
for f in os.listdir(test_img_dir):
    img = osp.join(test_img_dir, f)
    out_file = osp.join(out_img_dir, f)
    # 对单张图片进行推理并展示结果
    result = inference_segmentor(model, img)
    # 在新窗口中可视化推理结果
    #  model.show_result(img, result, show=True)
    # 或将可视化结果存储在文件中
    # 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
    model.show_result(img, result, out_file=out_file, opacity=1)
