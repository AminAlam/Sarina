import cv2 
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
import sys
from tqdm import tqdm
sys.path.append('./src/lib')
lib_cpp_backend = ctypes.cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

class DetectOuterBorder(object):
    def __init__(self):
        self.obj = lib_cpp_backend.DetectOuterBorder_c()

    def detect_border(self, array1, array2):
        border_x = np.empty_like(array1, dtype=np.uint16)
        border_y = np.empty_like(array2, dtype=np.uint16)
        func = lib_cpp_backend.DetectOuterBorder_func
        func.argtypes = [ctypes.c_int, ctl.ndpointer(np.uint16, flags='aligned, c_contiguous'), 
                        ctl.ndpointer(np.uint16, flags='aligned, c_contiguous'),  
                        ctypes.c_int, ctl.ndpointer(np.uint16, flags='aligned, c_contiguous'),
                        ctl.ndpointer(np.uint16, flags='aligned, c_contiguous')]
        func(self.obj, array1, array2, array1.shape[0], border_x, border_y)
        return border_x, border_y

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
    border_locations_x = border_locations_x.astype(np.uint16)
    border_locations_y = border_locations_y.astype(np.uint16)
    
    f = DetectOuterBorder()
    [border_x, border_y] = f.detect_border(border_locations_x, border_locations_y)
    border_x = border_x[border_x != 0]
    border_y = border_y[border_y != 0]
    x0 = border_x[0]
    y0 = border_y[0]
    x1 = border_x[1]
    y1 = border_y[1]
    area = 0
    thresh = border_x.shape[0]* 2/100
    for i in tqdm(range(1, border_x.shape[0], 1)):
        for ii in range(i+1, border_x.shape[0], 1):
            x0_tmp = border_x[i]
            y0_tmp = border_y[i]
            x1_tmp = border_x[ii]
            y1_tmp = border_y[ii]
            h_side = np.abs(float(y1_tmp) - float(y0_tmp))
            v_side = np.abs(float(x1_tmp) - float(x0_tmp))
            area_tmp = v_side*h_side
            rgb_img_copy = rgb_img.copy()
            if area_tmp>area:
                if x0_tmp<x1_tmp and y0_tmp<y1_tmp:
                    indexes = (border_x>=x0_tmp) & (border_x<=x1_tmp) & (border_y>=y0_tmp) & (border_y<=y1_tmp)
                elif x0_tmp<x1_tmp and y0_tmp>y1_tmp:
                    indexes = (border_x>=x0_tmp) & (border_x<=x1_tmp) & (border_y>=y1_tmp) & (border_y<=y0_tmp)
                elif x0_tmp>x1_tmp and y0_tmp<y1_tmp:
                    indexes = (border_x>=x1_tmp) & (border_x<=x0_tmp) & (border_y>=y0_tmp) & (border_y<=y1_tmp)
                elif x0_tmp>x1_tmp and y0_tmp>y1_tmp:
                    indexes = (border_x>=x1_tmp) & (border_x<=x0_tmp) & (border_y>=y1_tmp) & (border_y<=y0_tmp)
                x_tmp = border_x[indexes]
                y_tmp = border_y[indexes]
                if len(x_tmp) + len(y_tmp) < thresh:
                    x0 = x0_tmp
                    y0 = y0_tmp
                    x1 = x1_tmp
                    y1 = y1_tmp
                    area = area_tmp
    print(i, ii, x0, y0, x1, y1, area)

    for x,y in zip(border_x, border_y):
        cv2.circle(rgb_img, (y, x), 1, (0, 0, 255), 1)
    # cv2 rectangle
    cv2.rectangle(rgb_img, (y0, x0), (y1, x1), (0, 255, 0), 1)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)

