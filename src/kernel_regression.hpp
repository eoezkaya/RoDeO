#ifndef KERNELREG
#define KERNELREG

#include <armadillo>
#include "surrogate_model.hpp"
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


class KernelRegressionModel2 : public SurrogateModel {

private:

	mat mahalanobisMatrix;
	double sigmaGaussianKernel;

	mat mahalanobisMatrixAdjoint;
	double sigmaGaussianKernelAdjoint;

	mat lowerDiagonalMatrix;
	mat lowerDiagonalMatrixAdjoint;

	unsigned int maximumCrossValidationIterations;
	unsigned int maximumInnerOptIterations;
	unsigned int NvalidationSet;
	unsigned int Ntraining;


	mat dataTraining;
	mat	dataValidation;

	mat	XTraining;
	mat	XValidation;

	vec	yTraining;
	vec	yValidation;

	vec weightL12Regularization;
	LOSS_FUNCTION lossFunctionType;



public:

	KernelRegressionModel2();
	KernelRegressionModel2(std::string name, unsigned int dimension);

	void initializeSurrogateModel(void);
	void initializeMahalanobisMatrixRandom(void);
	void initializeSigmaRandom(void);


	void train(void);
	void calculateMahalanobisMatrix(void);
	void calculateMahalanobisMatrixAdjoint(void);
	void updateMahalanobisAndSigma(double learningRate);
	double calculateGaussianKernel(rowvec xi, rowvec xj) const;
	double calculateGaussianKernelAdjoint(rowvec xi, rowvec xj,double calculateGaussianKernelb);
	void calculateKernelValues(void);


	mat calculateKernelMatrix(void) const;
	void calculateKernelMatrixAdjoint(mat &kernelValuesMatrixb);
	mat calculateKernelRegressionWeights(mat kernelValuesMatrix) const;
	mat calculateKernelRegressionWeightsAdjoint(mat, mat &, mat) const;

	double calculateLossFunction(void);
	double calculateLossFunctionAdjoint(void);
	double calculateL1Loss(mat) const;
	double calculateL2Loss(mat) const;
	double calculateL1LossAdjoint(mat weights, mat &weightsb) const;
	double calculateL2LossAdjoint(mat weights, mat &weightsb) const;

	double calculateL12RegularizationTerm(double weight) const;
	double calculateL12RegularizationTermAdjoint(double weight);

	friend void testcalculateKernelRegressionWeightsAdjoint(void);
	friend void testcalculateKernelRegressionWeights(void);
	friend void testcalculateLossFunctionsAdjoint(void);
	friend void testcalculateGaussianKernelAdjoint(void);
	friend void testcalculateGaussianKernel(void);
	friend void testcalculateLossFunctionAdjointL2(void);
	friend void testcalculateLossFunctionAdjointL1(void);
	friend void testcalculateMahalanobisMatrix(void);


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


