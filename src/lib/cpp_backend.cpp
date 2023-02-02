#include <iostream>
#include <cmath>

class CppBackend{
    public:
        unsigned short int get_fontscale(unsigned short int *filled_area[][],
                                        const int min_x,
                                        const int min_y,
                                        const int max_x,
                                        const int max_y,
                                        const int w,
                                        const int h,
                                        const float weight,
                                        float fontScale_tmp,
                                        const float decay_rate,
                                        int x,
                                        int y){
            
            while (1){
                // x = random int between min_x and max_x
                x = min_x + (rand() % (max_x - min_x + 1));
                y = min_y + (rand() % (max_y - min_y + 1));
                // sum of filled_area
                int sum = 0;
                for (int i = x; i < x+h; i++){
                    for (int j = y; j < y+w; j++){
                        unsigned short int tmp = filled_area[i][j];
                        sum += tmp;
                    };
                };
                if (sum == 0){
                    break;
                };
            };
            
            
            return 0;
        }
};

extern "C" {
    CppBackend* CppBackend_c(){
        return new CppBackend();
    }
    unsigned short int get_fontscale_func(CppBackend* cppBackend, 
                                                unsigned short int *filled_area[][], 
                                                const int min_x,
                                                const int min_y,
                                                const int max_x,
                                                const int max_y,
                                                const int w,
                                                const int h,
                                                const float weight,
                                                float fontScale_tmp,
                                                const float decay_rate,
                                                int x,
                                                int y){ 
        return cppBackend->get_fontscale(filled_area, min_x, min_y, max_x, max_y, w, h, weight, fontScale_tmp, decay_rate, x, y); 
    }
};