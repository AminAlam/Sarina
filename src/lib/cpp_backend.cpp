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
                            double **filled_area,
                            unsigned short int *status,
                            const int margin = 20,
                            const int max_iter = 500){

            // status: 0 -> success, 1 -> fail
            
            int row_tmp;
            int col_tmp;
            int counter = 0;            
            while (1){
                col_tmp = min_x + (rand() % (max_x - min_x + 1));
                row_tmp = min_y + (rand() % (max_y - min_y + 1));
                double sum = 0;
                for (int i = row_tmp-margin; i < row_tmp+margin+h; i++){
                    if (sum > 0) break;
                    for (int j = col_tmp-margin; j < col_tmp+margin+w; j++){
                        sum += filled_area[i][j];
                        if (sum > 0) break;
                    };
                };
                if (sum == 0){
                    *status = 1;
                    break;
                }                  
                else if (counter > max_iter){
                    *status = 0;
                    break;
                };
                counter += 1;
            };

            *x = col_tmp;
            *y = row_tmp;
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
                                            double **filled_area,
                                            unsigned short int *status,
                                            const int margin = 20,
                                            const int max_iter = 500){
        return cppBackend->get_fontscale(min_x, min_y, max_x, max_y, w, h, x, y, filled_area, status, margin, max_iter); 
    }
};