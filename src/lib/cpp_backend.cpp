#include <iostream>
#include <cmath>

class DetectOuterBorder{
    public:
        unsigned short int detect_border(const unsigned short int *x_coords, const unsigned short int *y_coords, const int length){
            std::cout << "function 'detect_border' is called" << std::endl;

            unsigned short int threshold = 10;
            static unsigned short int* border_x = new unsigned short int[length];
            static unsigned short int* border_y = new unsigned short int[length];
            // initialize border_x and border_y
            for (int i = 0; i < length; i++){
                border_x[i] = 0;
                border_y[i] = 0;
            }

            for (int i = 0; i < length-1; i++){
                if(sqrt(pow(abs(x_coords[i] - x_coords[i+1]), 2) + pow(abs(y_coords[i] - y_coords[i+1]), 2)) < threshold){
                    border_x[i] = x_coords[i];
                    border_y[i] = y_coords[i];
                }
            }
            for (int i = 0; i < 10; i++){
                std::cout << border_x[i] << " " << border_y[i] << std::endl;
            }
            return *border_x;
        }
};

extern "C" {
    DetectOuterBorder* DetectOuterBorder_c(){
        return new DetectOuterBorder();
    }
    unsigned short int DetectOuterBorder_func(DetectOuterBorder* detectOuterBorder, const unsigned short int *x_coords, const unsigned short int *y_coords, const int length){ 
        return detectOuterBorder->detect_border(x_coords, y_coords, length); 
    }
};