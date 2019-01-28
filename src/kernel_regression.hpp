#ifndef KERNELREG_HPP
#define KERNELREG_HPP
#include <armadillo>
using namespace arma;



int trainMahalanobisDistance(mat &M, mat &data, double &sigma, double &wSvd, double &w12);


double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M);

#endif
