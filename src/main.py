import cv2 
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
import sys
sys.path.append('./src/lib')
lib_cpp_backend = ctypes.cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

# cv2.imshow('image', rgb_img)
# cv2.imshow('gray_img', gray_img)
# cv2.imshow('closing', closing)
# cv2.imshow('edges', image_edges)
# cv2.waitKey(0)


class DetectOuterBorder(object):
    def __init__(self):
        self.obj = lib_cpp_backend.DetectOuterBorder_c()

    def detect_border(self, array1, array2):
        length = array1.shape[0]
        func = lib_cpp_backend.DetectOuterBorder_func
        func.restype = ctl.ndpointer(dtype=np.uint16, shape=(length,), flags='aligned, c_contiguous')
        func.argtypes = [ctypes.c_int, ctl.ndpointer(np.uint16, flags='aligned, c_contiguous'), ctl.ndpointer(np.uint16, flags='aligned, c_contiguous'), ctypes.c_int]
        out = func(self.obj, array1, array2, array1.shape[0])
        print(out)
        out = np.ctypeslib.as_array((ctypes.c_uint16 * length).from_address(ctypes.addressof(out)))
        return out

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
    for x,y in zip(border_locations_x, border_locations_y):
        cv2.circle(rgb_img, (y, x), 1, (0, 0, 255), 1)
    f = DetectOuterBorder()
    out = f.detect_border(border_locations_x, border_locations_y)
    print(out)
