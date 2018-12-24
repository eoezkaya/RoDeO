#ifndef KERNELREG_HPP
#define KERNELREG_HPP
#include <armadillo>
using namespace arma;



int trainMahalanobisDistance(void);
double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M);

#endif
