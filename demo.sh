#!/bin/bash

python tools/test.py /home/b104/liao/code/open-mmlab/mmsegmentation/ex_res/ch03/iSAID/tmf-b/tchhead_swin_base_path4_window5_512x512_160k_isaid_pretrain_384x384.py /home/b104/liao/code/open-mmlab/mmsegmentation/ex_res/ch03/iSAID/tmf-b/iter_160000.pth --eval=mIoU,mFscore 
