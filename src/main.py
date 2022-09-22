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
counter = 0 


def detect_rects(border_x, border_y, x0, y0, x1, y1, rects_x_points=[], rects_y_points=[]):
    print('Detecting rectangles...')
    global counter
    counter = counter+1

    if counter>2:
        return [[0, 0, 0, 0]]

    rects = []
    area = 0
    thresh = border_x.shape[0]* 2/100
    print('thresh', thresh)
    if len(border_x) > 2000:
        step = 100
    elif len(border_x) > 1000:
        step = 50
    elif len(border_x) > 500:
        step = 10
    else:
        step = 1
    border_x = np.append(border_x, rects_x_points)
    border_y = np.append(border_y, rects_y_points)
    for i in range(1, border_x.shape[0], step):
        for ii in range(i+1, border_x.shape[0], step):
            x0_tmp = border_x[i]
            y0_tmp = border_y[i]
            x1_tmp = border_x[ii]
            y1_tmp = border_y[ii]
            h_side = np.abs(float(y1_tmp) - float(y0_tmp))
            v_side = np.abs(float(x1_tmp) - float(x0_tmp))
            area_tmp = v_side*h_side
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
                    print(counter, x0, y0, x1, y1, area)
    rects.append([x0, y0, x1, y1])
    if x0<=x1 and y0<=y1:
        x_small = x0
        x_big = x1
        y_small = y0
        y_big = y1
    elif x0<=x1 and y0>y1:
        x_small = x0
        x_big = x1
        y_small = y1
        y_big = y0
    elif x0>x1 and y0<=y1:
        x_small = x1
        x_big = x0
        y_small = y0
        y_big = y1
    elif x0>x1 and y0>y1:
        x_small = x1
        x_big = x0
        y_small = y1
        y_big = y0
    
    horizontal_top_y = np.arange(y_small, y_big, 1)
    horizontal_top_x = np.full_like(horizontal_top_y, x_small)

    horizontal_bottom_y = np.arange(y_small, y_big, 1)
    horizontal_bottom_x = np.full_like(horizontal_bottom_y, x_big)

    vertical_left_x = np.arange(x_small, x_big, 1)
    vertical_left_y = np.full_like(vertical_left_x, y_small)

    vertical_right_x = np.arange(x_small, x_big, 1)
    vertical_right_y = np.full_like(vertical_right_x, y_big)

    border_x = np.append(border_x, horizontal_bottom_x)
    border_y = np.append(border_y, horizontal_bottom_y)
    border_x = np.append(border_x, horizontal_top_x)
    border_y = np.append(border_y, horizontal_top_y)
    border_x = np.append(border_x, vertical_left_x)
    border_y = np.append(border_y, vertical_left_y)
    border_x = np.append(border_x, vertical_right_x)
    border_y = np.append(border_y, vertical_right_y)
    for i in range(len(vertical_left_x)):
        x = vertical_left_x[i]
        y = vertical_left_y[i]
        rect_points_y = np.arange(y, y_big)
        rect_points_x = np.full_like(rect_points_y, x)
        rects_x_points.append(rect_points_x)
        rects_y_points.append(rect_points_y)

    border_x = border_x.astype(np.uint16)
    border_y = border_y.astype(np.uint16)


    rects.extend(detect_rects(border_x, border_y, border_x[0], border_y[0], border_x[1], border_y[1], rects_x_points, rects_y_points))
    
    return rects

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
    rects = detect_rects(border_x, border_y, x0, y0, x1, y1)
    print(rects)
    for rect in rects:
        rect = [int(i) for i in rect]
        cv2.rectangle(rgb_img, (rect[1], rect[0]), (rect[3], rect[2]), (0, 0, 255), 1)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)

