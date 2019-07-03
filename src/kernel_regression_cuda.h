#ifndef KERNELREG_CUDA_H
#define KERNELREG_CUDA_H

#include <armadillo>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <codi.hpp>

using namespace arma;

#define numVar 13
#define number_of_threads_per_block 64

int trainMahalanobisDistance(mat &L, mat &data, double &sigma, double &wSvd, double &w12, int max_cv_iter);

double gaussianKernel(rowvec &xi, rowvec &xj, double sigma, mat &M);

double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma);

int calcRegTerms(double *L, double *regTerm, double wSvd, double w12, int dim);

double calcKernelValCPU(rowvec &xi, rowvec &xj, mat &M, double sigma);

void calcLossFunCPU(double *result, double *input, double *data, int N);

void calcLossFunGPU(double *result, double *input, double *data,int N);

#endif
