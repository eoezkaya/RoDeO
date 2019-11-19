#ifndef KERNELREG
#define KERNELREG

#include <armadillo>

using namespace arma;


int trainMahalanobisDistance(fmat &L, fmat &data, float &sigma, float &wSvd, float &w12,
		int max_cv_iter, int lossFunType, int batchsize, int nepochs);

int trainMahalanobisDistanceBruteForce(fmat &L, fmat &data, float &sigma, float ymax, int lossFunType, int batchsize, int ntrials);


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


