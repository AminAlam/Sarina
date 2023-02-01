from traceback import print_tb
import cv2 
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
import sys
from tqdm import tqdm
import random

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


def extract_text(file_path):
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        lines = [line.decode('utf-8') for line in lines]
    weights = [line.split(' ')[-1] for line in lines]
    # remove any character that is not a number from weights
    weights = [float(''.join([char for char in weight if char.isdigit() or char == '.'])) for weight in weights]
    lines = [' '.join(line.split(' ')[:-1]) for line in lines]
    # remove the last tabs and new lines
    lines = [line.strip() for line in lines]
    return lines, weights


if __name__ == "__main__":
    txt_file = 'assets/texts/merged.txt'
    img_file = 'assets/images/iran_map.png'
    rgb_img = cv2.imread(img_file)
    text, weights = extract_text(txt_file)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contour = contours[1]
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

    

    contour[:, :, 0] -= x_rect
    contour[:, :, 1] -= y_rect
    rgb_img = np.zeros((h_rect, w_rect, 3), dtype=np.uint8)

    font = cv2.QT_FONT_BLACK
    fontScale = 0.15
    color = (255, 0, 255)
    thickness = 2
    txt_info = []
    # get area of the contour 

    S_bounding_rect = cv2.contourArea(contour)
    S_total_txt = 0
    for indx, txt in enumerate(text):
        (w, h), _ = cv2.getTextSize(txt, font, int(fontScale*weights[indx]), thickness)
        S_total_txt += w * h
        txt_info.append([w, h, indx, txt, weights[indx]])

    txt_info.sort(key=lambda x: x[-1], reverse=True)
    C_ratio = S_bounding_rect / S_total_txt
    # epsilon = 0.002 * cv2.arcLength(contour, True)
    # contour = cv2.approxPolyDP(contour, epsilon, True)
    
    contour = contour * int(1/C_ratio**0.5)
    contour = contour.astype(np.int32)
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

    print(cv2.contourArea(contour)/S_total_txt)
    # resize rgb_img 
    rgb_img = cv2.resize(rgb_img, (contour[:, :, 0].max(), contour[:, :, 1].max()))
    rgb_img_tmp = rgb_img.copy()
    cv2.drawContours(rgb_img, [contour], 0, (0, 255, 0), 1)


    # find center of contour
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    min_x = contour[:, :, 0].min()
    max_x = contour[:, :, 0].max()
    min_y = contour[:, :, 1].min()
    max_y = contour[:, :, 1].max()
    cv2.circle(rgb_img, (cx, cy), 5, (0, 0, 255), -1)
    filled_area = np.zeros((h_rect, w_rect), dtype=np.uint8)
    cv2.drawContours(filled_area, [contour], 0, (255, 255, 255), -1)
    filled_area = (filled_area / 255 - 1)*-1
    decay_rate = 0.9
    # put text in txt_info in the contour center
    for txt in txt_info:
        fontScale_tmp = fontScale
        w, h, indx, txt, weight = txt
        # get a random point in the contour
        counter = 0
        while True:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            if filled_area[y:y+h, x:x+w].sum() == 0:
                break            
            if counter > 1000:
                fontScale_tmp = fontScale_tmp * decay_rate
                (w, h), _ = cv2.getTextSize(txt, font, int(fontScale_tmp*weight), thickness)
                counter = 0
            counter += 1

        
        cv2.putText(rgb_img, txt, (x, y+h), font, int(fontScale_tmp*weight), color, thickness, cv2.LINE_AA)
        cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(filled_area, txt, (x, y+h), font, int(fontScale_tmp*weight), color, thickness, cv2.LINE_AA)
        # filled_area[y:y+h, x:x+w] = 1

        cv2.imshow('filled_area', filled_area*255)
        cv2.imshow('rgb_img', rgb_img)
        cv2.waitKey(1)
        

    # save rgb_img
    cv2.imwrite('results/iran_map_with_text.png', rgb_img)
    filled_area = (filled_area / 255 - 1)*-1
    cv2.imwrite('results/iran_map_filled_area.png', filled_area*255)

    