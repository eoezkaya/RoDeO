#ifndef KERNELREG_HPP
#define KERNELREG_HPP
#include <armadillo>
using namespace arma;



int trainMahalanobisDistance(mat &M,mat &X,vec &ys,double &sigma);


double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M);

#endif
