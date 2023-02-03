#include <iostream>
#include <cmath>
#include <vector>
#include "ndarray.h"

class CppBackend{
    public:
        int * get_fontscale(const int min_x,
                                        const int min_y,
                                        const int max_x,
                                        const int max_y,
                                        const int w,
                                        const int h,
                                        const float weight,
                                        float fontScale_tmp,
                                        const float decay_rate,
                                        int* x,
                                        int* y,
                                        unsigned short int **filled_area){
            int x_tmp;
            int y_tmp;            
            while (1){
                
                x_tmp = min_x + (rand() % (max_x - min_x + 1));
                y_tmp = min_y + (rand() % (max_y - min_y + 1));

                unsigned int sum = 0;
                for (int i = x_tmp; i < x_tmp+h; i++){
                    for (int j = y_tmp; j < y_tmp+w; j++){
                        sum += filled_area[i][j];
                    };
                };
                if (sum == 0){
                    break;
                };
            };

            *x = x_tmp;
            *y = y_tmp;

            std::cout << *x << ' ' << *y << std::endl;
            
            return x, y;
        }
};

extern "C" {
    CppBackend* CppBackend_c(){
        return new CppBackend();
    }
     int * get_fontscale_func(CppBackend* cppBackend, 
                                                const int min_x,
                                                const int min_y,
                                                const int max_x,
                                                const int max_y,
                                                const int w,
                                                const int h,
                                                const float weight,
                                                float fontScale_tmp,
                                                const float decay_rate,
                                                int* x,
                                                int* y,
                                                unsigned short int **filled_area){
        return cppBackend->get_fontscale(min_x, min_y, max_x, max_y, w, h, weight, fontScale_tmp, decay_rate, x, y, filled_area); 
    }
};