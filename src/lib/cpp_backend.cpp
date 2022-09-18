#include <iostream>
#include "ndarray.h"
#include <cmath>

class DetectOuterBorder{
    public:
        int detect_border(numpyArray<int> array1, numpyArray<int> array2){
            std::cout << "function 'detect_border' is called" << std::endl;
            Ndarray<int,2> x_coords(array1);
            Ndarray<int,2> y_coords(array2);
            int sum=0;
            return sum;
        }
};

extern "C" {
    DetectOuterBorder* DetectOuterBorder_c(){
        return new DetectOuterBorder();
    }
    int DetectOuterBorder_func(DetectOuterBorder* detectOuterBorder, numpyArray<int> array1, numpyArray<int> array2){ 
        return detectOuterBorder->detect_border(array1, array2); 
    }
};