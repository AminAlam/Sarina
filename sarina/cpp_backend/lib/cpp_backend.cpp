#include <iostream>
#include <cmath>
#include <vector>

class CppBackend{
    public:
        void find_txt_position(const int min_x,
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

            int row_start = 0;
            int row_end = 0;
            int col_start = 0;
            int col_end = 0;
            
            int counter = 0;            
            while (1){
                col_tmp = min_x + (rand() % (max_x - min_x + 1));
                row_tmp = min_y + (rand() % (max_y - min_y + 1));

                if (row_tmp < margin) row_start = row_tmp;
                else row_start = row_tmp - margin;
                if (row_tmp + margin + h > max_y) row_end = max_y;
                else row_end = row_tmp + margin + h;
                if (col_tmp < margin) col_start = row_tmp;
                else col_start = col_tmp - margin;
                if (col_tmp + margin + w > max_x) col_end = max_x;
                else col_end = col_tmp + margin + w;
                
                double sum = 0;
                for (int i = row_start; i < row_end; i++){
                    if (sum > 0) break;
                    for (int j = col_start; j < col_end; j++){
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
     void find_txt_position_func(CppBackend* cppBackend, 
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
        return cppBackend->find_txt_position(min_x, min_y, max_x, max_y, w, h, x, y, filled_area, status, margin, max_iter); 
    }
};