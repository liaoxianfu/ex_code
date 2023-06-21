from libtiff import TIFF
import cv2

import PIL.Image as Image
tif = Image.open('top_potsdam_4_15_RGB.tif', mode='r')

tif.save("aa.png")