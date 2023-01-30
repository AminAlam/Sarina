#include <iostream>
#include <cmath>

class DetectOuterBorder{
    public:
        unsigned short int detect_border(const unsigned short int *x_coords, 
                                        const unsigned short int *y_coords, 
                                        const int length, 
                                        unsigned short int *border_x, 
                                        unsigned short int *border_y){

            std::cout << "function 'detect_border' is called" << std::endl;
            unsigned short int threshold = 10;
            unsigned short int distance = 0;
            for (int i = 0; i < length; i++){
                border_x[i] = 0;
                border_y[i] = 0;
                distance = sqrt(pow(x_coords[i+1] - x_coords[i], 2) + pow(y_coords[i+1] - y_coords[i], 2));
                if(distance < threshold){
                    border_x[i] = x_coords[i];
                    border_y[i] = y_coords[i];
                }
            }

            
            return *border_x, *border_y;
        }
};

extern "C" {
    DetectOuterBorder* DetectOuterBorder_c(){
        return new DetectOuterBorder();
    }
    unsigned short int DetectOuterBorder_func(DetectOuterBorder* detectOuterBorder, 
                                                const unsigned short int *x_coords, 
                                                const unsigned short int *y_coords, 
                                                const int length, 
                                                unsigned short int *border_x, 
                                                unsigned short int *border_y){ 
        return detectOuterBorder->detect_border(x_coords, y_coords, length, border_x, border_y); 
    }
};