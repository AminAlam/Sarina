#include <iostream>
#include <cmath>
#include <vector>

class CppBackend{
    public:
        void get_fontscale(const int min_x,
                            const int min_y,
                            const int max_x,
                            const int max_y,
                            int w,
                            int h,
                            int* x,
                            int* y,
                            unsigned short int **filled_area,
                            unsigned short int *status,
                            const int margin = 20,
                            const int max_iter = 500){

            // status: 0 -> success, 1 -> fail
            
            int x_tmp;
            int y_tmp;
            int counter = 0;            

            while (1){
                    
                y_tmp = min_x + (rand() % (max_x - min_x + 1));
                x_tmp = min_y + (rand() % (max_y - min_y + 1));

                // calculate the sum of filled_area[y_tmp-margin:y_tmp+margin+h, x_tmp-margin:x_tmp+margin+w] in a fast way

                unsigned int sum = 0;
                for (int i = y_tmp-margin; i < y_tmp+margin+h; i++){
                    if (sum > 0) break;
                    for (int j = x_tmp-margin; j < x_tmp+margin+w; j++){
                        sum += filled_area[i][j];
                        if (sum > 0) break;
                    };
                };
                if (sum == 0){
                    *status = 1;
                    break;}                  
                else if (counter > max_iter)
                {
                    *status = 0;
                    break;
                };
                counter += 1;
            };

            *x = x_tmp;
            *y = y_tmp;
        };
};

extern "C" {
    CppBackend* CppBackend_c(){
        return new CppBackend();
    }
     void get_fontscale_func(CppBackend* cppBackend, 
                                            const int min_x,
                                            const int min_y,
                                            const int max_x,
                                            const int max_y,
                                            int w,
                                            int h,
                                            int* x,
                                            int* y,
                                            unsigned short int **filled_area,
                                            unsigned short int *status,
                                            const int margin = 20,
                                            const int max_iter = 500){
        return cppBackend->get_fontscale(min_x, min_y, max_x, max_y, w, h, x, y, filled_area, status, margin, max_iter); 
    }
};