#include <iostream>

class DetectOuterBorder{
    public:
        void detect_border(){
            std::cout << "Hello 1111" << std::endl;
        }
};

extern "C" {
    DetectOuterBorder* DetectOuterBorder_c(){ return new DetectOuterBorder(); }
    void DetectOuterBorder_func(DetectOuterBorder* detectOuterBorder){ detectOuterBorder->detect_border(); }
};
