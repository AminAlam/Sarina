import cv2 
import numpy as np
import ctypes as ct
import sys
from tqdm import tqdm
import random 

sys.path.append('./src/lib')
lib_cpp_backend = ct.cdll.LoadLibrary('./src/lib/lib_cpp_backend.so')

class CppBackend(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.obj = lib_cpp_backend.CppBackend_c()
        self.get_fontscale_func = lib_cpp_backend.get_fontscale_func
        self.x = 0
        self.y = 0
        self.min_x = ct.c_int(min_x)
        self.min_y = ct.c_int(min_y)
        self.max_x = ct.c_int(max_x)
        self.max_y = ct.c_int(max_y)

    def get_fontScale(self, filled_area, w, h, max_iter, margin):
        
        status = 0
        max_iter = ct.c_int(max_iter)
        margin = ct.c_int(margin)
        w = ct.c_int(w)
        h = ct.c_int(h)

        UI16Ptr = ct.POINTER(ct.c_uint16)
        INTPtr = ct.POINTER(ct.c_int)
        FLOAT64Ptr = ct.POINTER(ct.c_double)        
        FLOAT64PtrPtr = ct.POINTER(FLOAT64Ptr)
        
        ct_arr = np.ctypeslib.as_ctypes(filled_area)
        UI16PtrArr = UI16Ptr * ct_arr._length_
        ct_ptr_to_filled_area = ct.cast(UI16PtrArr(*(ct.cast(row, UI16Ptr) for row in ct_arr)), FLOAT64PtrPtr)

        ct_ptr_to_x = ct.cast(ct.pointer(ct.c_int(self.x)), INTPtr)
        ct_ptr_to_y = ct.cast(ct.pointer(ct.c_int(self.y)), INTPtr)
        ct_ptr_to_status = ct.cast(ct.pointer(ct.c_int(status)), UI16Ptr)

        self.get_fontscale_func.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, 
                                            ct.c_int, ct.c_int, INTPtr, INTPtr, 
                                            FLOAT64PtrPtr, UI16Ptr, ct.c_int, ct.c_int]
        self.get_fontscale_func(self.obj, self.min_x, self.min_y, self.max_x, self.max_y, w, h,
                                 ct_ptr_to_x, ct_ptr_to_y, ct_ptr_to_filled_area, ct_ptr_to_status, margin, max_iter)

        x = ct_ptr_to_x.contents.value
        y = ct_ptr_to_y.contents.value
        status = ct_ptr_to_status.contents.value

        return x, y, status


def extract_text(file_path):
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        lines = [line.decode('utf-8') for line in lines]
    confs = [line.split('|')[-1] for line in lines]    
    weights = [float(''.join([char for char in conf if char.isdigit() or char == '.'])) for conf in confs]
    lines = ['|'.join(line.split('|')[:-1]) for line in lines]
    lines = [line.strip() for line in lines]
    return lines, weights


if __name__ == "__main__":
    txt_file = 'assets/texts/merged.txt'
    img_file = 'assets/images/iran_map.png'
    rgb_img = cv2.imread(img_file)
    original_img = rgb_img.copy()
    main_img = rgb_img.copy()
    text, weights = extract_text(txt_file)

    w_img, h_img, _ = rgb_img.shape

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # for i in range(2):
    #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     cv2.drawContours(rgb_img, contours, i, color, 3)
    #     cv2.putText(rgb_img, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 2)
    # cv2.imshow('contours', rgb_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    keep_contours = [1]
    contour = np.concatenate(([contours[i] for i in keep_contours]), axis=0)
    
    rgb_img = (rgb_img*0+1)*255

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 0.15
    thickness_original = 10
    decay_rate = 0.9
    max_iter_original = 1000
    margin = np.min([20, w_img//100, h_img//100])


    txt_info = []

    S_contour = cv2.contourArea(contour)

    max_weight = max(weights)
    min_weight = min(weights)
    max_min_ratio = max_weight / min_weight

    weights = [30 + 170 * (weight - min_weight) / (max_weight - min_weight) for weight in weights]
    max_weight = max(weights)
    min_weight = min(weights)
    max_min_ratio = max_weight / min_weight

    S_total_txt = 0
    for indx, txt in enumerate(text):
        weight = weights[indx]
        thickness = 1+int(thickness_original*(max_weight/weight)**0.5)
        (w, h), _ = cv2.getTextSize(txt, font, int(1+fontScale*weights[indx]), thickness)
        S_total_txt += w * h
        txt_info.append([w, h, indx, txt, weights[indx]])

    txt_info.sort(key=lambda x: x[-1], reverse=True)
    C_ratio = S_contour / S_total_txt
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    resize_factor = 1/C_ratio**0.5
    contour = contour * resize_factor
    contour = contour.astype(np.int32)
    rgb_img = cv2.resize(rgb_img, (0, 0), fx=resize_factor, fy=resize_factor)
    main_img = cv2.resize(main_img, (0, 0), fx=resize_factor, fy=resize_factor)

    min_x = contour[:, :].min()
    max_x = contour[:, :].max()
    min_y = contour[:, :].min()
    max_y = contour[:, :].max()
    filled_area = rgb_img*0
    filled_area = cv2.cvtColor(filled_area, cv2.COLOR_BGR2GRAY)
    for contour_indx in keep_contours:
        contours[contour_indx] = contours[contour_indx] * resize_factor
        contours[contour_indx] = contours[contour_indx].astype(np.int32)
        cv2.drawContours(filled_area, contours, contour_indx, (255, 255, 255), -1)

    filled_area = (filled_area / 255 - 1)*-1

    max_weight = np.max(weights)
    cpp_backend = CppBackend(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
    for txt in tqdm(txt_info):
        w, h, indx, txt, weight = txt
        max_iter = int(max_iter_original*(max_weight/weight)**0.5)
        thickness = 1+int(thickness_original/(max_weight/weight)**0.5)
        fontScale_tmp = fontScale
        while True:
            x, y, status = cpp_backend.get_fontScale(filled_area, w, h, max_iter, margin)
            if status:
                (w, h), _ = cv2.getTextSize(txt, font, fontScale_tmp*weight, thickness)
                break
            else:
                fontScale_tmp = fontScale_tmp * decay_rate
                (w, h), _ = cv2.getTextSize(txt, font, fontScale_tmp*weight, thickness)

        alpha = 0.5*(1+weight/max_weight)
        color = (0,0,0)

        overlay = rgb_img.copy()
        cv2.putText(overlay, txt, (x, y+h), font, fontScale_tmp*weight, color, thickness, cv2.LINE_AA)
        rgb_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)
        cv2.putText(main_img, txt, (x, y+h), font, fontScale_tmp*weight, color, thickness, cv2.LINE_AA)
        cv2.putText(filled_area, txt, (x, y+h), font, fontScale_tmp*weight, [255, 255, 255], thickness, cv2.LINE_AA)
        
    alpha = 0.3
    overlay = rgb_img.copy()
    for contour_indx in keep_contours:
        cv2.drawContours(main_img, contours, contour_indx, (0, 0, 0), 10)

    main_img = cv2.resize(main_img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    cv2.imwrite('results/iran_map_with_text.png', rgb_img)
    cv2.imwrite('results/iran_map_filled_area.png', filled_area*255)
    cv2.imwrite('results/iran_map_with_text_and_contour.png', main_img)

    