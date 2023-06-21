import os
import os.path as osp
import shutil

dataset = 'data/potsdam'
img_dir = 'img_dir/val'
gt_dir = 'potsdam_val'
org_img_dir = osp.join(dataset, img_dir)
gt_img_dir = osp.join(dataset, gt_dir)
target_dir = 'log/tmfb_potsdam/img'
target_org_dir = osp.join(target_dir, 'org')
target_gt_dir = osp.join(target_dir, 'gt')
if not osp.exists(target_org_dir):
    os.makedirs(target_org_dir)
if not osp.exists(target_gt_dir):
    os.makedirs(target_gt_dir)

img_name_list = [
    '2_13_0_0_512_512.png', '2_13_2560_1536_3072_2048.png', '2_13_5120_2048_5632_2560.png', '2_14_0_3072_512_3584.png',
    '2_14_1536_5488_2048_6000.png', '3_13_3072_3072_3584_3584.png', '3_13_4096_5120_4608_5632.png',
    '3_13_5120_1024_5632_1536.png', '3_13_5488_1536_6000_2048.png', '4_13_2048_2048_2560_2560.png',
    '4_14_3072_3584_3584_4096.png'
]
gt_name_list = img_name_list

for img in img_name_list:
    src_img = osp.join(org_img_dir, img)
    target_img = osp.join(target_org_dir, img)
    print("copy file {} to {}".format(src_img, target_img))
    shutil.copyfile(src_img, target_img)

for gt in gt_name_list:
    src_gt = osp.join(gt_img_dir, gt)
    target_gt = osp.join(target_gt_dir, gt)
    print("copy file {} to {}".format(src_gt, target_gt))
    shutil.copyfile(src_gt, target_gt)
