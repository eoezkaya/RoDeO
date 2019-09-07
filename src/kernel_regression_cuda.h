#ifndef KERNELREG_CUDA_H
#define KERNELREG_CUDA_H

#include <armadillo>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <codi.hpp>

using namespace arma;

#define numVar 38
#define number_of_threads_per_block 64

int trainMahalanobisDistance(fmat &L, fmat &data, float &sigma, float &wSvd, float &w12,int max_cv_iter, int lossFunType);

int trainMahalanobisDistanceWithGradient(fmat &L, fmat &data, float &sigma, float &wSvd, float &w12,int max_cv_iter, int lossFunType);

float gaussianKernel(frowvec &xi, frowvec &xj, float sigma, fmat &M);

float kernelRegressor(fmat &X, fvec &y, frowvec &xp, fmat &M, float sigma);
float kernelRegressor(fmat &X, fvec &y, fmat &grad, frowvec &xp, fmat &M, float sigma) ;


float kernelRegressorNotNormalized(fmat &X,
								   fmat &XnotNormalized,
								   fvec &y,
								   fmat &grad,
								   frowvec &xp,
								   fvec &xmin,
								   fvec &xmax,
								   fmat &M,
								   float sigma);

double kernelRegressorNotNormalized(mat &X,
		mat &XnotNormalized,
		vec &y,
		mat &grad,
		rowvec &xp,
		vec &xmin,
		vec &xmax,
		mat &M,
		double sigma);


#endif
