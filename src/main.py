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
    confs = [line.split('|')[-1] for line in lines]    
    weights = [float(''.join([char for char in conf if char.isdigit() or char == '.'])) for conf in confs]
    lines = ['|'.join(line.split('|')[:-1]) for line in lines]
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
    rgb_img = np.ones((h_rect, w_rect, 3), dtype=np.uint8)*255

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 0.15
    thickness = 10
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
    
    contour = contour * 1/C_ratio**0.5
    contour = contour.astype(np.int32)
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

    # resize rgb_img 
    rgb_img = cv2.resize(rgb_img, (contour[:, :, 0].max(), contour[:, :, 1].max()))


    # find center of contour
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    min_x = contour[:, :, 0].min()
    max_x = contour[:, :, 0].max()
    min_y = contour[:, :, 1].min()
    max_y = contour[:, :, 1].max()
    filled_area = np.zeros((h_rect, w_rect), dtype=np.uint8)
    cv2.drawContours(filled_area, [contour], 0, (255, 255, 255), -1)

    filled_area = (filled_area / 255 - 1)*-1
    decay_rate = 0.95
    max_weight = np.max(weights)
    # put text in txt_info in the contour center
    for txt in tqdm(txt_info):
        fontScale_tmp = fontScale
        w, h, indx, txt, weight = txt
        # get a random point in the contour
        counter = 0
        while True:
            x = random.randint(min_x+20, max_x-20)
            y = random.randint(min_y+20, max_y-20)            
            if filled_area[y-18:y+h+18, x-18:x+w+18].sum() == 0:
                break            
            if counter == 200:
                fontScale_tmp = fontScale_tmp * decay_rate
                (w, h), _ = cv2.getTextSize(txt, font, int(fontScale_tmp*weight), thickness)
                counter = 0
            counter += 1

        alpha = 0.5*(1+weight/max_weight)
        # put text in the image with alpha 0.5
        color = (0,0,0)
        if txt == 'Sarina':
            color = (0,0,255)
            thickness = 20
        else:
            thickness = 10
        overlay = rgb_img.copy()
        cv2.putText(overlay, txt, (x, y+h), font, int(fontScale_tmp*weight), color, thickness, cv2.LINE_AA)
        rgb_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)

        cv2.putText(filled_area, txt, (x, y+h), font, int(fontScale_tmp*weight), [255, 255, 255], thickness, cv2.LINE_AA)
        # filled_area[y:y+h, x:x+w] = 1
    alpha = 0.3
    overlay = rgb_img.copy()
    # draw the border of the border of the contour
    cv2.drawContours(overlay, [contour], 0, (0,0,255), 10)
    rgb_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)
    # save rgb_img
    cv2.imwrite('results/iran_map_with_text.png', rgb_img)
    cv2.imwrite('results/iran_map_filled_area.png', filled_area*255)

    