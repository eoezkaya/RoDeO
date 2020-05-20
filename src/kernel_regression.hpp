#ifndef KERNELREG
#define KERNELREG

#include <armadillo>

using namespace arma;


class KernelRegressionModel {

public:

	unsigned int dim;
	unsigned int N;
	std::string label;
	std::string kernelRegHyperParamFilename;

	mat M;
	mat data;
	mat X;
	double sigma;

	KernelRegressionModel(std::string name,int dimension);

};




int trainMahalanobisDistance(mat &L, mat &data, double &sigma, double &wSvd, double &w12,
		unsigned int max_cv_iter, unsigned int lossFunType, unsigned int batchsize, unsigned int nepochs);



double gaussianKernel(rowvec &xi, rowvec &xj, double sigma, mat &M);

double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma);
double kernelRegressor(mat &X, vec &y, mat &grad, rowvec &xp, mat &M, double sigma) ;


double kernelRegressorNotNormalized(mat &X,
								   mat &XnotNormalized,
								   vec &y,
								   mat &grad,
								   rowvec &xp,
								   vec &xmin,
								   vec &xmax,
								   mat &M,
								   double sigma);

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


