#ifndef KERNELREG_CUDA_H
#define KERNELREG_CUDA_H

#include <armadillo>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <codi.hpp>

using namespace arma;

#define numVar 10
#define number_of_threads_per_block 64

int trainMahalanobisDistance(fmat &L, fmat &data, float &sigma, float &wSvd, float &w12, int max_cv_iter);

float gaussianKernel(frowvec &xi, frowvec &xj, float sigma, fmat &M);

float kernelRegressor(fmat &X, fvec &y, frowvec &xp, fmat &M, float sigma);


#endif
