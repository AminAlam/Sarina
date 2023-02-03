import cv2 
import numpy as np
import numpy.ctypeslib as ctl
import ctypes as ct
import sys
from tqdm import tqdm
import random

sys.path.append('./src/lib')
lib_cpp_backend = ct.cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

class CppBackend(object):
    def __init__(self):
        self.obj = lib_cpp_backend.CppBackend_c()
        self.get_fontscale_func = lib_cpp_backend.get_fontscale_func

    def get_fontScale(self, filled_area, min_x, min_y, max_x, max_y, w, h, weight, fontScale_tmp, decay_rate):
        x = 0
        y = 0
        filled_area = filled_area.astype(np.uint16)
        min_x = int(min_x)
        min_y = int(min_y)
        max_x = int(max_x)
        max_y = int(max_y)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        weight = float(weight)
        fontScale_tmp = float(fontScale_tmp)
        decay_rate = float(decay_rate)

        ct_arr = np.ctypeslib.as_ctypes(filled_area)
        UI16Ptr = ct.POINTER(ct.c_uint16)
        INTPtr = ct.POINTER(ct.c_int)
        UI16PtrPtr = ct.POINTER(UI16Ptr)
        UI16PtrArr = UI16Ptr * ct_arr._length_
        ct_ptr_to_filled_area = ct.cast(UI16PtrArr(*(ct.cast(row, UI16Ptr) for row in ct_arr)), UI16PtrPtr)
        ct_ptr_to_x = ct.cast(ct.pointer(ct.c_int(x)), INTPtr)
        ct_ptr_to_y = ct.cast(ct.pointer(ct.c_int(y)), INTPtr)

        self.get_fontscale_func.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, 
                        ct.c_int, ct.c_float, ct.c_float, ct.c_float, INTPtr, INTPtr,
                        UI16PtrPtr]
        self.get_fontscale_func(self.obj, min_x, min_y, max_x, max_y, w, h, weight, fontScale_tmp, decay_rate, ct_ptr_to_x, ct_ptr_to_y, ct_ptr_to_filled_area)
        print('x: ', x, 'y: ', y)
        return fontScale_tmp, x, y


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
    cpp_backend = CppBackend()
    txt_file = 'assets/texts/merged.txt'
    img_file = 'assets/images/Maryam.png'
    rgb_img = cv2.imread(img_file)
    original_img = rgb_img.copy()
    main_img = rgb_img.copy()
    text, weights = extract_text(txt_file)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # for i in range(5):
    #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     cv2.drawContours(rgb_img, contours, i, color, 3)
    #     cv2.putText(rgb_img, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 2)
    # cv2.imshow('contours', rgb_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    keep_contours = [0]
    contour = np.concatenate((contours[0]), axis=0)
    

    # contour[:, :, 0] -= x_rect
    # contour[:, :, 1] -= y_rect
    rgb_img = (rgb_img*0+1)*255

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 0.15
    thickness = 10
    txt_info = []

    S_contour = cv2.contourArea(contour)
    S_total_txt = 0
    for indx, txt in enumerate(text):
        (w, h), _ = cv2.getTextSize(txt, font, int(fontScale*weights[indx]), thickness)
        S_total_txt += w * h
        txt_info.append([w, h, indx, txt, weights[indx]])

    txt_info.sort(key=lambda x: x[-1], reverse=True)
    C_ratio = S_contour / S_total_txt
    # epsilon = 0.002 * cv2.arcLength(contour, True)
    # contour = cv2.approxPolyDP(contour, epsilon, True)
    
    resize_factor = 1/C_ratio**0.5
    contour = contour * resize_factor
    contour = contour.astype(np.int32)

    # resize rgb_img with resize_factor
    rgb_img = cv2.resize(rgb_img, (0, 0), fx=resize_factor, fy=resize_factor)
    main_img = cv2.resize(main_img, (0, 0), fx=resize_factor, fy=resize_factor)

    # find center of contour
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    min_x = contour[:, :].min()
    max_x = contour[:, :].max()
    min_y = contour[:, :].min()
    max_y = contour[:, :].max()
    filled_area = rgb_img*0
    # convert filled_area to gray
    filled_area = cv2.cvtColor(filled_area, cv2.COLOR_BGR2GRAY)
    for contour_indx in keep_contours:
        contours[contour_indx] = contours[contour_indx] * resize_factor
        contours[contour_indx] = contours[contour_indx].astype(np.int32)
        cv2.drawContours(filled_area, contours, contour_indx, (255, 255, 255), -1)

    filled_area = (filled_area / 255 - 1)*-1
    # cv2.imshow('filled_area', filled_area)
    # cv2.waitKey(0)
    decay_rate = 0.95
    max_weight = np.max(weights)
    # put text in txt_info in the contour center
    for txt in tqdm(txt_info):
        w, h, indx, txt, weight = txt
        # get a random point in the contour
        # min_x, min_y, w, h, weight, fontScale_tmp, decay_rate
        counter = 0
        fontScale_tmp, x, y = cpp_backend.get_fontScale(filled_area, min_x, min_y, max_x, max_y, w, h, weight, fontScale, decay_rate)
        # while True:
        #     x = random.randint(min_x+20, max_x-20)
        #     y = random.randint(min_y+20, max_y-20)            
        #     if filled_area[y-18:y+h+18, x-18:x+w+18].sum() == 0:
        #         break            
        #     if counter == 200:
        #         fontScale_tmp = fontScale_tmp * decay_rate
        #         # (w, h), _ = cv2.getTextSize(txt, font, int(fontScale_tmp*weight), thickness)
        #         w = int(w * decay_rate**0.5)
        #         h = int(h * decay_rate**0.5)
        #         counter = 0
        #     counter += 1

        alpha = 0.5*(1+weight/max_weight)
        # put text in the image with alpha 0.5
        color = (0,0,0)
        if txt == 'Sarina':
            color = (0,0,0)
            thickness = 60
        else:
            thickness = 20
        overlay = rgb_img.copy()
        cv2.putText(overlay, txt, (x, y+h), font, int(fontScale_tmp*weight), color, thickness, cv2.LINE_AA)
        rgb_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)
        cv2.putText(main_img, txt, (x, y+h), font, int(fontScale_tmp*weight), color, thickness, cv2.LINE_AA)

        cv2.putText(filled_area, txt, (x, y+h), font, int(fontScale_tmp*weight), [255, 255, 255], thickness, cv2.LINE_AA)
        # filled_area[y:y+h, x:x+w] = 1
    alpha = 0.3
    overlay = rgb_img.copy()
    # draw the border of the border of the contour
    for contour_indx in keep_contours:
        cv2.drawContours(main_img, contours, contour_indx, (0, 0, 0), 10)

    # resize main_imf to original_img
    main_img = cv2.resize(main_img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    cv2.imwrite('results/iran_map_with_text.png', rgb_img)
    cv2.imwrite('results/iran_map_filled_area.png', filled_area*255)
    cv2.imwrite('results/iran_map_with_text_and_contour.png', main_img)

    