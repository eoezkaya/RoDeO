#ifndef SVM_HPP
#define SVM_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
using namespace arma;

class SVM_param {
public:

    SVM_KERNEL_TYPE kernel_type;
    double sv_tolerance;

    SVM_param (){

        kernel_type = SVM_LINEAR;
        sv_tolerance = 10E-6;
    }

    void print(void){

    }
} ;


void train_svm(mat &X, vec &y,vec &alpha, double &b, SVM_param model_parameters);

#endif
