#ifndef TRAIN_RBF_HPP
#define TRAIN_RBF_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
using namespace arma;


class RBF_param {
public:
    int max_number_of_iter_for_cv;
    int number_of_cv_inner_iter;
    double upper_bound_sigma;
    double lower_bound_sigma;
    double upper_bound_lambda;
    double lower_bound_lambda;
    int number_of_data_points_for_cross_val;
    RBF_TYPE type;

    RBF_param (){

        max_number_of_iter_for_cv = 120;
        number_of_cv_inner_iter = 20;
        upper_bound_sigma = 5.0;
        lower_bound_sigma = 0.0;
        upper_bound_lambda= 12;
        lower_bound_lambda= 6;
        type = LINEAR;
    }

    void print(void){
        printf("number of iterations for cross validation = %d\n",max_number_of_iter_for_cv);
        printf("number of inner iterations for cross validation = %d\n",number_of_cv_inner_iter);
        printf("lower bound for sigma = %15.10f\n",lower_bound_sigma);
        printf("upper bound for sigma = %15.10f\n",upper_bound_sigma);
        printf("lower bound for lambda = %15.10f\n",lower_bound_lambda);
        printf("upper bound for lambda = %15.10f\n",upper_bound_lambda);

    }
} ;



int train_rbf(mat &X, vec &ys, vec &w, double &sigma, RBF_param model_parameters);
double calc_ftilde_rbf(mat &X, rowvec &xp, vec &w, int type, double sigma);

#endif
