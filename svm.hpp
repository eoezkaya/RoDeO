#ifndef SVM_HPP
#define SVM_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
using namespace arma;

class SVM_param {
public:

    SVM_KERNEL_TYPE kernel_type;

    SVM_param (){

        kernel_type = SVM_LINEAR;
    }

    void print(void){

    }
} ;


void train_svm(mat &X, vec &y,SVM_param model_parameters);

#endif
