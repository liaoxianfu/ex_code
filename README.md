### ch2测试



```python
python tools/test.py ex_res/ch03/iSAID/tmf-b/tchhead_swin_base_path4_window5_512x512_160k_isaid_pretrain_384x384.py ex_res/ch03/iSAID/tmf-b/iter_160000.pth --eval 'mIoU' 'mFscore' 
```



```python
python tools/test.py ex_res/ch03/loveda/tmf-b/baseline_lr1e-5_upernet_swin_small_patch4_window7_512x512_160k_loveDA_pretrain_224x224_1K.py ex_res/ch03/loveda/tmf-b/iter_160000.pth --eval 'mIoU' 'mFscore' 
```



```python
python tools/test.py ex_res/ch03/loveda/tmf-s/baseline_lr1e-5_upernet_swin_base_path4_window5_512x512_160k_loveDA_pretain_384x384_22k.py ex_res/ch03/loveda/tmf-t/iter_160000.pth --eval 'mIoU' 'mFscore' 
```



```python
python tools/test.py ex_res/ch03/loveda/tmf-t/baseline_lr1e-5_upernet_swin_tiny_patch4_window7_512x512_160k_loveDA_pretrain_224x224_1K.py ex_res/ch03/loveda/tmf-t/iter_160000.pth --eval 'mIoU' 'mFscore'
```



```python
python tools/test.py ex_res/ch03/Potsdam/tmf-b/baseline_lr1e-5_upernet_swin_base_path4_window5_512x512_160k_potsdam_pretain_384x384_22k.py ex_res/ch03/Potsdam/tmf-b/iter_160000.pth --eval  'mFscore' 
```



```python
python tools/test.py ex_res/ch03/Potsdam/tmf-s/baseline_lr1e-5_upernet_swin_small_path4_window7_512x512_160k_potsdam_pretrain_224x224_1K.py ex_res/ch03/Potsdam/tmf-s/iter_160000.pth --eval  'mFscore' 
```



```python
python tools/test.py ex_res/ch03/Potsdam/tmf-t/ baseline_lr1e-5_upernet_swin_tiny_path4_window7_512x512_160k_potsdam_pretrain_224x224_1K.py ex_res/ch03/Potsdam/tmf-t/iter_160000.pth --eval  'mFscore' 
```





### ch3测试



```python
python tools/test.py ex_res/ch04/iSAID/tmf-b/loss_isaid_swin_conv6_lr1e-5_upernet_swin_conv6_base_path4_window5_512x512_160k_isaid_pretain_384x384_22k.py ex_res/ch04/iSAID/tmf-b/iter_320000.pth --eval 'mIoU' 'mFscore' 
```



```python
python tools/test.py ex_res/ch04/loveda/base/loss_loveda_swin_base_conv6_160k.py ex_res/ch04/loveda/base/iter_80000.pth --eval 'mIoU' 'mFscore'
```



```python
python tools/test.py ex_res/ch04/loveda/small/loss_loveda_swin_small_conv6_160k.py ex_res/ch04/loveda/small/iter_80000.pth --eval 'mIoU' 'mFscore' 
```



```python
python tools/test.py ex_res/ch04/loveda/tiny/ loss_loveda_swin_tiny_conv6_160k.py ex_res/ch04/loveda/small/iter_144000.pth --eval 'mIoU' 'mFscore' 
```





```python
python tools/test.py ex_res/ch04/potsdam/base/loss_potsdam_swin_base_conv6_160k.py ex_res/ch04/potsdam/base/last.pth --eval  'mFscore' 
```



```python
python tools/test.py ex_res/ch04/potsdam/small/ loss_potsdam_swin_small_conv6_160k.py
ex_res/ch04/potsdam/small/iter_160000.pth --eval  'mFscore' 
```



```python
python tools/test.py ex_res/ch04/potsdam/tiny/loss_potsdam_swin_tiny_conv6_160k.py ex_res/ch04/potsdam/tiny/iter_160000.pth --eval  'mFscore' 
```

### ch4测试

```python
python tools/test.py ex_res/ch5/potsdam/potsdam_lightdw_mobilevit_v2_200.py  ex_res/ch5/potsdam/iter_144000.pth --eval 'mIoU' 'mFscore' 
```



```python
 python tools/test.py ex_res/ch5/vaihingen/vaihingen_lightdw_mobilevit_v2_100.py ex_res/ch5/vaihingen/iter_128000.pth  --eval 'mIoU' 'mFscore' 
```





训练配置文件在b104服务器`/home/b104/liao/code/open-mmlab/mmsegmentation/local_configs`路径下,模型代码`/home/b104/liao/code/open-mmlab/mmsegmentation/mmseg`下，模型改动方法请参考https://mmsegmentation.readthedocs.io/en/latest/ 

