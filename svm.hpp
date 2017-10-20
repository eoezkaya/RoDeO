#ifndef SVM_HPP
#define SVM_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "test_functions.hpp"

using namespace arma;

class SVM_param {
public:

    SVM_KERNEL_TYPE kernel_type;
    double sv_tolerance;
    int max_inner_iter;
    double ppolykernel;
    double sigmaGausskernel;
    double cpolykernel;
    double Csoftmargin;

    SVM_param (){

        kernel_type = SVM_LINEAR;
        sv_tolerance = 10E-6;
        max_inner_iter = 10;
        ppolykernel=2.0;
        sigmaGausskernel=1.0;
        cpolykernel=1.0;
        Csoftmargin=1014;
    }

    void print(void){

        printf("kernel type = %d\n",kernel_type);
        printf("max_inner_iter = %d\n",max_inner_iter);
        printf("p(polynomial kernel) = %10.7e\n",ppolykernel);
        printf("c(polynomial kernel) = %10.7e\n",cpolykernel);
        printf("sigma(Gaussian kernel) = %10.7e\n",sigmaGausskernel);
        printf("C (soft margin parameters) = %10.7e\n",Csoftmargin);


    }
} ;


void train_svm(mat &X, vec &y,vec &alpha, double &b, SVM_param &model_parameters);
double svm_linear_kernel(rowvec x1, rowvec x2);
double svm_polynomial_kernel(rowvec x1, rowvec x2, double c, double p);
double svm_rbf_kernel(rowvec x1, rowvec x2, double sigma);

void perform_svn_test(Classifier_Function_param &test_function,
        int  number_of_samples,
        int number_of_outliers,
        SVM_KERNEL_TYPE kernel_type);

#endif
