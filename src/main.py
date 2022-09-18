import cv2 
import numpy as np
from ctypes import cdll
import sys
sys.path.append('./src/lib')
from numpyctypes import c_ndarray
lib_cpp_backend = cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

# cv2.imshow('image', rgb_img)
# cv2.imshow('gray_img', gray_img)
# cv2.imshow('closing', closing)
# cv2.imshow('edges', image_edges)
# cv2.waitKey(0)


class DetectOuterBorder(object):
    def __init__(self):
        self.obj = lib_cpp_backend.DetectOuterBorder_c()

    def detect_border(self, array1, array2):
        return lib_cpp_backend.DetectOuterBorder_func(self.obj, array1, array2)

if __name__ == "__main__":

    img_file = 'assets/images/iran_map.png'
    rgb_img = cv2.imread(img_file)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    heigth, width = gray_img.shape
    gray_img = 255 - gray_img
    img_mean = np.mean(gray_img)
    img_std = np.std(gray_img)
    gray_img[gray_img > img_mean+img_std] = 255
    gray_img[gray_img <= img_mean+img_std] = 0

    closing = gray_img
    image_edges = cv2.Canny(closing, 10, 20)
    [border_locations_x, border_locations_y] = np.where(image_edges == 255)
    for x,y in zip(border_locations_x, border_locations_y):
        # cv2 add circle to image
        cv2.circle(rgb_img, (y, x), 1, (0, 0, 255), 1)
    image_edges = image_edges*0
    arg1 = c_ndarray(border_locations_x, dtype=border_locations_x.dtype, ndim = len(border_locations_x.shape), shape = tuple(border_locations_x.shape))
    arg2 = c_ndarray(border_locations_y, dtype=border_locations_y.dtype, ndim = len(border_locations_y.shape), shape = tuple(border_locations_y.shape))
    f = DetectOuterBorder()
    out = f.detect_border(arg1, arg2)
    print(out)
