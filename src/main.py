from traceback import print_tb
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


def detect_rects(border_x, border_y, x0, y0, x1, y1, rgb_img):
    print('Detecting rectangles...')
    point_a_indx = 0
    point_b_indx = 0
    rects = []
    area = 0
    
    thresh = 10
    print('thresh', thresh)
    if len(border_x) > 2000:
        step = 100
    elif len(border_x) > 1000:
        step = 50
    elif len(border_x) > 500:
        step = 10
    else:
        step = 1
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
                    point_a_indx = i
                    point_b_indx = ii
                    area = area_tmp
                    print(x0, y0, x1, y1, area)
    rects.append([x0, y0, x1, y1])
    print(point_a_indx, point_b_indx)
    if point_a_indx < point_b_indx:
        sec1_indexes = list(range(point_a_indx, point_b_indx))
        border_x_sec1 = border_x[sec1_indexes]
        border_y_sec1 = border_y[sec1_indexes]
        sec2_indexes = list(range(point_b_indx, border_x.shape[0]))
        sec2_indexes.extend(range(0, point_a_indx))
        border_x_sec2 = border_x[sec2_indexes]
        border_y_sec2 = border_y[sec2_indexes]
    else:
        sec1_indexes = list(range(point_b_indx, point_a_indx))
        border_x_sec1 = border_x[sec1_indexes]
        border_y_sec1 = border_y[sec1_indexes]
        sec2_indexes = list(range(point_a_indx, border_x.shape[0]))
        sec2_indexes.extend(range(0, point_b_indx))
        border_x_sec2 = border_x[sec2_indexes]
        border_y_sec2 = border_y[sec2_indexes]

    if x0<x1 and y0<y1:
        border_x_sec1 = np.append(border_x_sec1, np.arange(x0, x1))
        border_y_sec1 = np.append(border_y_sec1, np.full_like(np.arange(x0, x1), y1))
        border_y_sec1 = np.append(border_y_sec1, np.arange(y0, y1))
        border_x_sec1 = np.append(border_x_sec1, np.full_like(np.arange(y0, y1), x0))
    elif x0<x1 and y0>y1:
        border_y_sec1 = np.append(border_y_sec1, np.arange(y1, y0))
        border_x_sec1 = np.append(border_x_sec1, np.full_like(np.arange(y1, y0), x1))
        border_x_sec1 = np.append(border_x_sec1, np.arange(x0, x1))
        border_y_sec1 = np.append(border_y_sec1, np.full_like(np.arange(x0, x1), y0))
    elif x0>x1 and y0<y1:
        

    
    

    cv2.rectangle(rgb_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    for x,y in zip(border_x_sec1, border_y_sec1):
        cv2.circle(rgb_img, (x, y), 1, (255, 255, 0), 1)
        cv2.imshow('rgb_img', rgb_img)
        cv2.waitKey(1)    
    return rects

if __name__ == "__main__":

    img_file = 'assets/images/iran_map.png'
    rgb_img = cv2.imread(img_file)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = contours[1]
    border_locations_x = [i[0][0] for i in contours]
    border_locations_y = [i[0][1] for i in contours]
    border_locations_x = np.array(border_locations_x, dtype=np.uint16)
    border_locations_y = np.array(border_locations_y, dtype=np.uint16)
    x0 = border_locations_x[0]
    y0 = border_locations_y[0]
    x1 = border_locations_x[1]
    y1 = border_locations_y[1]
    rects = detect_rects(border_locations_x, border_locations_y, x0, y0, x1, y1, rgb_img)
    print(rects)
