/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */

#ifndef KERNELREG
#define KERNELREG

#include <armadillo>
#include "surrogate_model.hpp"
using namespace arma;


class KernelRegressionModel : public SurrogateModel {

private:

	mat mahalanobisMatrix;
	double sigmaGaussianKernel;

	mat mahalanobisMatrixAdjoint;
	double sigmaGaussianKernelAdjoint;

	mat lowerDiagonalMatrix;
	mat lowerDiagonalMatrixAdjoint;



	PartitionData testDataForInnerOptimizationLoop;
	PartitionData trainingData;

	vec weightL12Regularization;
	LOSS_FUNCTION lossFunctionType;



public:

	unsigned int maximumCrossValidationIterations;
	unsigned int maximumInnerOptIterations;

	KernelRegressionModel();
	KernelRegressionModel(std::string name);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolate(rowvec x) const ;
	double interpolateWithGradients(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;
	double calculateInSampleError(void) const;


	void setGradientsOn(void);
	void setGradientsOff(void);

	void initializeMahalanobisMatrixRandom(void);
	void initializeSigmaRandom(void);

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
	friend void testcalculateLossFunctions(void);


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


