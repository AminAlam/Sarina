import cv2 
import numpy as np
from ctypes import cdll
lib_cpp_backend = cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

# img_file = 'assets/images/iran_map.png'
# rgb_img = cv2.imread(img_file)

# gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
# heigth, width = gray_img.shape
# gray_img = 255 - gray_img
# img_mean = np.mean(gray_img)
# img_std = np.std(gray_img)
# gray_img[gray_img > img_mean+img_std] = 255
# gray_img[gray_img <= img_mean+img_std] = 0

# closing = gray_img
# image_edges = cv2.Canny(closing, 10, 20)
# [border_locations_x, border_locations_y] = np.where(image_edges == 255)
# for x,y in zip(border_locations_x, border_locations_y):
#     # cv2 add circle to image
#     cv2.circle(rgb_img, (y, x), 1, (0, 0, 255), 1)
# cv2.imshow('image', rgb_img)
# cv2.imshow('gray_img', gray_img)
# cv2.imshow('closing', closing)
# cv2.imshow('edges', image_edges)
# cv2.waitKey(0)


class DetectOuterBorder(object):
    def __init__(self):
        self.obj = lib_cpp_backend.DetectOuterBorder_c()

    def detect_border(self):
        lib_cpp_backend.DetectOuterBorder_func(self.obj)

if __name__ == "__main__":
    f = DetectOuterBorder()
    f.detect_border() 