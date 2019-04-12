#ifndef KERNELREG_HPP
#define KERNELREG_HPP
#include <armadillo>
using namespace arma;



int trainMahalanobisDistance(mat &M, mat &data, double &sigma, double &wSvd, double &w12,int max_cv_iter=0);


double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma);


double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M);



double computeGenErrorKernelReg(double (*test_function)(double *),
		double *bounds,
		int dim,
		int number_of_samples,
		mat &X,
		vec &y,
		mat &M,
		vec &x_min,
		vec &x_max,
		double sigma);

#endif
