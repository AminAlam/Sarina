import ctypes as ct
import numpy as np
import sys
import os
import pathlib

file_dir = pathlib.Path(__file__).parent.absolute()
lib_cpp_backend = ct.cdll.LoadLibrary(os.path.join(file_dir, 'lib', 'lib_cpp_backend.so'))

class CppBackend(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.obj = lib_cpp_backend.CppBackend_c()
        self.find_txt_position_func = lib_cpp_backend.find_txt_position_func
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

        self.find_txt_position_func.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, 
                                            ct.c_int, ct.c_int, INTPtr, INTPtr, 
                                            FLOAT64PtrPtr, UI16Ptr, ct.c_int, ct.c_int]
        self.find_txt_position_func(self.obj, self.min_x, self.min_y, self.max_x, self.max_y, w, h,
                                 ct_ptr_to_x, ct_ptr_to_y, ct_ptr_to_filled_area, ct_ptr_to_status, margin, max_iter)

        x = ct_ptr_to_x.contents.value
        y = ct_ptr_to_y.contents.value
        status = ct_ptr_to_status.contents.value

        return x, y, status