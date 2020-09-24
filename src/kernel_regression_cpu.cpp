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
#include <math.h>
#include <armadillo>
#include <iostream>
#include <stack>

#include "kernel_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"

using namespace std;
using namespace arma;




KernelRegressionModel::KernelRegressionModel():SurrogateModel(){}


KernelRegressionModel::KernelRegressionModel(std::string name):SurrogateModel(name),trainingData("TrainingData"),testDataForInnerOptimizationLoop("CrossValidationData"){

	modelID = KERNEL_REGRESSION;
	hyperparameters_filename = label + "_kernel_regression_hyperparameters.csv";

}

void KernelRegressionModel::initializeSurrogateModel(void){

	if(label != "None"){

		cout << "Initializing kernel regression model...\n";

		ReadDataAndNormalize();

		mahalanobisMatrix = zeros<mat>(dim,dim);
		mahalanobisMatrixAdjoint = zeros<mat>(dim,dim);

		lowerDiagonalMatrix = zeros<mat>(dim,dim);
		lowerDiagonalMatrixAdjoint = zeros<mat>(dim,dim);
		sigmaGaussianKernel = 0.0;
		sigmaGaussianKernelAdjoint = 0.0;

		maximumCrossValidationIterations = 2;
		maximumInnerOptIterations = 1000;

		lossFunctionType = L2_LOSS_FUNCTION;

		weightL12Regularization = zeros<vec>(maximumCrossValidationIterations);



		int NvalidationSet = N/5;
		int Ntraining = N - NvalidationSet;

		/* divide data into training and validation data, validation data is used for optimizing regularization parameters */

		mat shuffledrawData = rawData;
		if(NvalidationSet > 0){

			shuffledrawData = shuffle(rawData);
		}

		mat dataTraining   = shuffledrawData.submat( 0, 0, Ntraining-1, dim );
		dataTraining.col(dim) = (dataTraining.col(dim) - ymin)/(ymax -ymin);

		trainingData.fillWithData(dataTraining);
		trainingData.normalizeAndScaleData(xmin,xmax);

		if(NvalidationSet > 0){

			mat dataValidation    = shuffledrawData.submat( Ntraining, 0, N-1, dim );
			dataValidation.col(dim) = (dataValidation.col(dim) - ymin)/(ymax -ymin);
			testDataForInnerOptimizationLoop.fillWithData(dataValidation);
			testDataForInnerOptimizationLoop.normalizeAndScaleData(xmin,xmax);


		}

	}

	for(unsigned int i=0; i<maximumCrossValidationIterations; i++){

		weightL12Regularization(i) = pow(10.0,generateRandomDouble(-5,0.0));
	}


	sigmaGaussianKernel = 1.0;

	lowerDiagonalMatrix.eye();
	mahalanobisMatrix =  lowerDiagonalMatrix * trans(lowerDiagonalMatrix);

	ifInitialized = true;

	cout << "Kernel regression model initialization is done...\n";
}

void KernelRegressionModel::saveHyperParameters(void) const{

	vec saveBuffer(dim*dim+1, fill::zeros);

	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++) saveBuffer(i*dim+j) = mahalanobisMatrix(i,j);

	saveBuffer(dim*dim) = sigmaGaussianKernel;

	saveBuffer.save(hyperparameters_filename, csv_ascii);
}

void KernelRegressionModel::loadHyperParameters(void){

	vec loadBuffer;
	loadBuffer.load(hyperparameters_filename, csv_ascii);

	assert(loadBuffer.size() == dim*dim+1);


	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++) mahalanobisMatrix(i,j) = loadBuffer(i*dim+j);

	sigmaGaussianKernel = loadBuffer(dim*dim);


}


void KernelRegressionModel::printSurrogateModel(void) const{

	cout << "\nKernel regression model:\n";
	cout << "N = " << N <<"\n";;
	cout << "dim = " << dim <<"\n";
	printMatrix(rawData,"rawData");
	printMatrix(X,"X");
	trainingData.print();
	testDataForInnerOptimizationLoop.print();

	printHyperParameters();

}

void KernelRegressionModel::printHyperParameters(void) const{

	printMatrix(mahalanobisMatrix,"mahalanobisMatrix");
	cout << "sigma = " <<sigmaGaussianKernel << "\n";
}





/* This function initializes sigmaGaussianKernel in a special way:
 * First we try different values of sigma randomly and find the best value that
 * maximizes the standard deviation of the kernel values of a sample.
 * In the second step, we add a small random deviation to sigma to have randomness
 *
 *
 */
void KernelRegressionModel::initializeSigmaRandom(void){

	double mean = 0.5;
	double deviation = 0.0;
	int Ntraining = trainingData.numberOfSamples;

	unsigned int nTrials = 100;
	vec kernelValues(Ntraining, fill::zeros);

	double maxstandardDeviation = 0.0;
	double bestSigma = 0.0;

	if(testDataForInnerOptimizationLoop.numberOfSamples > 0){

		for(unsigned int i=0; i<nTrials; i++){


			sigmaGaussianKernel = pow(10, generateRandomDouble(-2.0,1.0));

			rowvec x = testDataForInnerOptimizationLoop.getRow(0);

			for(unsigned j=0; j<Ntraining; j++) {

				rowvec xj = trainingData.getRow(j);
				kernelValues(j) = calculateGaussianKernel(x,xj);

			}
#if 0
			printVector(kernelValues, "kernelValues");
#endif
			double standardDeviation = stddev(kernelValues);
#if 0
			cout <<"standardDeviation: " <<standardDeviation << std::endl;
#endif
			if(standardDeviation > maxstandardDeviation){
				maxstandardDeviation =  standardDeviation;
				bestSigma = sigmaGaussianKernel;


			}

		}
		mean = bestSigma;

#if 0
		cout << "Best value of sigma = " << bestSigma << "\n";
		cout << "Standard deviation = " << maxstandardDeviation << "\n";
#endif

	}

	deviation = mean * 0.1;
	sigmaGaussianKernel = mean+ generateRandomDouble(-deviation,deviation);

}




void KernelRegressionModel::initializeMahalanobisMatrixRandom(void){


	for (unsigned int i = 0; i < dim; i++) {

		for (unsigned int j = 0; j <= i; j++) {

			if(i == j) { /* main diagonal */

				lowerDiagonalMatrix(i,j) = 1.0+ generateRandomDouble(-0.01,0.01);
			}
			else {

				lowerDiagonalMatrix(i,j) = generateRandomDouble(0.0,0.1);
			}
		}
	}

	calculateMahalanobisMatrix();


}


void KernelRegressionModel::calculateMahalanobisMatrix(void){

	mat LT(dim,dim, fill::zeros);
	mahalanobisMatrix.fill(0.0);

	for(unsigned int i=0; i<dim; i++){

		for(unsigned int j=0; j<dim; j++){

			if(lowerDiagonalMatrix(i,j) < 0){

				cout << "ERROR: entries of the lowerDiagonalMatrix cannot be negative!\n";
				printMatrix(lowerDiagonalMatrix,"lowerDiagonalMatrix");
				abort();

			}
		}
	}



	for(unsigned int i=0; i<dim; i++){

		for(unsigned int j=i; j<dim; j++){

			LT(i,j) = lowerDiagonalMatrix(j,i);
		}
	}

	/* Multiplying matrix L and LT and storing in M */
	for(unsigned int i = 0; i < dim; ++i)
		for(unsigned int j = 0; j < dim; ++j)
			for(unsigned int k = 0; k < dim; ++k)
			{
				mahalanobisMatrix(i,j) += lowerDiagonalMatrix(i,k) * LT(k,j);

			}

}


mat KernelRegressionModel::getMahalanobisMatrix(void) const{

	return mahalanobisMatrix;


}

double KernelRegressionModel::getsigmaGaussianKernel(void) const{

	return sigmaGaussianKernel;

}

void KernelRegressionModel::calculateMahalanobisMatrixAdjoint(void) {


	mat LT(dim,dim, fill::zeros);
	mat LTb(dim,dim, fill::zeros);

	for(unsigned int i=0; i<dim; i++)

		for(unsigned int j=i; j<dim; j++){

			LT(i,j) = lowerDiagonalMatrix(j,i);
		}


	for (int i = int(dim)-1; i > -1; --i)
		for (int j = int(dim)-1; j > -1; --j)
			for (int k = int(dim)-1; k > -1; --k) {
				lowerDiagonalMatrixAdjoint(i,k) += LT(k,j)*mahalanobisMatrixAdjoint(i,j);
				LTb(k,j) += lowerDiagonalMatrix(i,k)*mahalanobisMatrixAdjoint(i,j);
			}


	for (int i = int(dim)-1; i > -1; --i)
		for (int j = int(dim)-1; j > i-1; --j) {
			lowerDiagonalMatrixAdjoint(j,i) +=  LTb(i,j);
			LTb(i,j) = 0.0;
		}



}





void KernelRegressionModel::updateMahalanobisAndSigma(double learningRate){


	for(unsigned int i=0; i<dim; i++)

		for(unsigned int j=0; j<=i; j++){

			lowerDiagonalMatrix(i,j) = lowerDiagonalMatrix(i,j) - learningRate*lowerDiagonalMatrixAdjoint(i,j);

			if(lowerDiagonalMatrix(i,j) < 0.0) {

				lowerDiagonalMatrix(i,j) = 10E-06;
			}


		}

	sigmaGaussianKernel = sigmaGaussianKernel - learningRate*0.01* sigmaGaussianKernelAdjoint;

	if(sigmaGaussianKernel < 0.0){

		sigmaGaussianKernel = 10E-06;
	}

	calculateMahalanobisMatrix();

}

void KernelRegressionModel::train(void){

	if(!ifInitialized){

		printf("ERROR: Kernel regression model must be initialized before training!\n");
		abort();

	}


	/* we use only training samples for the interpolations */
	mat Xsave = X;
	X = trainingData.X;

	vec ysave = y;
	y = trainingData.yExact;

	const double learningRate = 0.00001;

	mat bestMOuterLoop;
	double bestSigmaOuterLoop;
	double bestMSE = LARGE;

	for(unsigned int iterCV=0; iterCV< maximumCrossValidationIterations; iterCV++){

		double weightL12 =  weightL12Regularization(iterCV);

#if 1
		printf("\n\nOuter iteration = %d\n",iterCV);
		printf("weight for the regularization term = %10.7f\n",weightL12);
#endif

		initializeMahalanobisMatrixRandom();
		initializeSigmaRandom();

		mat bestM;
		double bestSigma;


		weightL12 = 0.0;
		/* optimization loop */

		double objFunInitiaValue = 0.0;
		double objFunBest = LARGE;

		for(unsigned int iterInnerOpt=0 ; iterInnerOpt < maximumInnerOptIterations; iterInnerOpt++){



			double lossFuncValue = calculateLossFunctionAdjoint();
			double regularizationTerm = calculateL12RegularizationTermAdjoint(weightL12);
			double objFun = lossFuncValue +  regularizationTerm;

			if(iterInnerOpt == 0){

				objFunInitiaValue = objFun;

			}

			if(iterInnerOpt %100 == 0){
				double percentIterationsCompleted = ((iterInnerOpt*100.0)/maximumInnerOptIterations);
				cout <<"\r" <<"iterations completed = %" << percentIterationsCompleted << flush;

			}
#if 0
			if(iterInnerOpt %100 == 0){

				cout <<"objFun = "<<objFun <<"\n";
				cout << "normalized descent in objective function value = "<<objFun/objFunInitiaValue<<"\n";


				if(dim < 10){

					printMatrix(mahalanobisMatrix,"mahalanobisMatrix");
					cout << "sigma = " << sigmaGaussianKernel <<"\n";
				}

			}
#endif
			if(objFun < objFunBest){

				objFunBest = objFun;
				bestM = mahalanobisMatrix;
				bestSigma = sigmaGaussianKernel;

			}


			updateMahalanobisAndSigma(learningRate);

		} /* inner optimization loop */
#if 1
		cout << "\nnormalized descent in objective function value = "<<objFunBest/objFunInitiaValue<<"\n";
#endif
		mahalanobisMatrix = bestM;
		sigmaGaussianKernel = bestSigma;

		tryModelOnTestSet(testDataForInnerOptimizationLoop);

		double MSEInnerOptLoop = testDataForInnerOptimizationLoop.calculateMeanSquaredError();

#if 1
		cout <<"MSE = "<< MSEInnerOptLoop <<"\n";
#endif
		if(MSEInnerOptLoop < bestMSE){

			bestMSE = MSEInnerOptLoop;
			bestMOuterLoop = mahalanobisMatrix;
			bestSigmaOuterLoop = sigmaGaussianKernel;


		}


	} /* outer optimization loop */

	mahalanobisMatrix = bestMOuterLoop;
	sigmaGaussianKernel = bestSigmaOuterLoop;

	printHyperParameters();

	/* we retrieve full data for the offline interpolations */
	X = Xsave;
	y = ysave;

	saveHyperParameters();


}


double KernelRegressionModel::calculateLossFunction(void){

	if(!ifInitialized){

		std::cout <<"\nERROR: calculateLossFunction cannot be called without model initialization!\n";
		abort();
	}

	calculateMahalanobisMatrix();

	assert(sigmaGaussianKernel > 0);
	assert(mahalanobisMatrix.is_sympd());

	double lossFunctionValue = 0.0;
	mat kernelValuesMatrix = calculateKernelMatrix();
#if 0
	printMatrix(kernelValuesMatrix,"kernelValuesMatrix");
#endif
	mat regressionWeightsMatrix = calculateKernelRegressionWeights(kernelValuesMatrix);
#if 0
	printMatrix(regressionWeightsMatrix,"regressionWeightsMatrix");
#endif
	switch ( lossFunctionType )
	{
	case L1_LOSS_FUNCTION:
		lossFunctionValue = calculateL1Loss(regressionWeightsMatrix);
		break;
	case L2_LOSS_FUNCTION:
		lossFunctionValue = calculateL2Loss(regressionWeightsMatrix);
		break;
	default:
		printf("Error: Unknown lossFunType at %s, line %d\n",__FILE__, __LINE__);
		abort();
	}


	return lossFunctionValue;


}

double KernelRegressionModel::calculateLossFunctionAdjoint(void){

	int Ntraining = trainingData.numberOfSamples;
	calculateMahalanobisMatrix();

	assert(sigmaGaussianKernel > 0);


	double lossFunctionValue = 0.0;
	mat kernelValuesMatrix = calculateKernelMatrix();

	mat regressionWeightsMatrix = calculateKernelRegressionWeights(kernelValuesMatrix);

	mat regressionWeightsMatrixAdj(Ntraining,Ntraining,fill::zeros);
	mat kernelValuesMatrixAdj(Ntraining,Ntraining,fill::zeros);

	switch ( lossFunctionType )
	{
	case L1_LOSS_FUNCTION:
		lossFunctionValue = calculateL1LossAdjoint(regressionWeightsMatrix,regressionWeightsMatrixAdj);
		break;
	case L2_LOSS_FUNCTION:
		lossFunctionValue = calculateL2LossAdjoint(regressionWeightsMatrix,regressionWeightsMatrixAdj);
		break;
	default:
		printf("Error: Unknown lossFunType at %s, line %d\n",__FILE__, __LINE__);
		exit(-1);
	}

	regressionWeightsMatrix = calculateKernelRegressionWeightsAdjoint(kernelValuesMatrix,kernelValuesMatrixAdj,regressionWeightsMatrixAdj);

	calculateKernelMatrixAdjoint(kernelValuesMatrixAdj);

	calculateMahalanobisMatrixAdjoint();

	return lossFunctionValue;


}





mat KernelRegressionModel::calculateKernelRegressionWeightsAdjoint(mat kernelValuesMatrix, mat &kernelValuesMatrixb, mat kernelWeightMatrixb) const {

	int Ntraining = trainingData.numberOfSamples;
	assert(Ntraining >0);

	mat kernelWeightMatrix(Ntraining,Ntraining,fill::zeros);

	vec kernelSum(Ntraining, fill::zeros);
	vec kernelSumb(Ntraining, fill::zeros);


	for (unsigned int i = 0; i < Ntraining; ++i) {
		for (unsigned int j = 0; j < Ntraining; ++j) {
			if (i != j) {

				kernelSum(i) = kernelSum(i) + kernelValuesMatrix(i,j);

			}
		}


		for (unsigned int j = 0; j < Ntraining; ++j) {

			if (i != j) {

				kernelWeightMatrix(i,j) =  kernelValuesMatrix(i,j)/kernelSum(i);
			}

		}
	}

	for (int i = Ntraining-1; i > -1; --i) {
		{
			double tempb;
			for (int j = Ntraining-1; j > -1; --j) {

				if (i != j) {
					tempb = kernelWeightMatrixb(i,j)/kernelSum(i);
					kernelWeightMatrixb(i,j) = 0.0;
					kernelValuesMatrixb(i,j) += tempb;
					kernelSumb(i) = kernelSumb(i) - kernelValuesMatrix(i,j)*tempb/kernelSum(i);
				}
			}
		}
		for (int j = Ntraining-1; j > -1; --j) {
			if (i != j) {

				kernelValuesMatrixb(i,j) = kernelValuesMatrixb(i,j) +kernelSumb(i);
			}
		}
	}

	return kernelWeightMatrix;
}





mat KernelRegressionModel::calculateKernelRegressionWeights(mat kernelValuesMatrix) const{

	int Ntraining = trainingData.numberOfSamples;
	assert(Ntraining >0);
	assert(kernelValuesMatrix.is_symmetric());

	mat kernelWeightMatrix = zeros<mat>(Ntraining,Ntraining);

	vec kernelSum = zeros<vec>(Ntraining);

	for(unsigned int i=0; i<Ntraining; i++){

		for(unsigned int j=0; j<Ntraining; j++){

			if(i!=j) kernelSum(i) +=  kernelValuesMatrix(i,j);
		}

		for(unsigned int j=0; j<Ntraining; j++){

			if(i!=j) kernelWeightMatrix(i,j) =  kernelValuesMatrix(i,j)/kernelSum(i);
		}
	}

	return kernelWeightMatrix;
}


void KernelRegressionModel::calculateKernelMatrixAdjoint(mat &kernelValuesMatrixb) {

	int Ntraining = trainingData.numberOfSamples;
	sigmaGaussianKernelAdjoint = 0.0;
	mahalanobisMatrixAdjoint.fill(0.0);

	for (int i = Ntraining-1; i > -1; --i) {
		double resb;
		double tmpb;

		for (int j = Ntraining-1; j > i-1; --j) {

			rowvec xi = trainingData.getRow(i);
			rowvec xj = trainingData.getRow(j);

			tmpb = kernelValuesMatrixb(j,i);
			kernelValuesMatrixb(j,i) = 0.0;
			kernelValuesMatrixb(i,j) += tmpb;
			resb = kernelValuesMatrixb(i,j);
			kernelValuesMatrixb(i,j) = 0.0;
			calculateGaussianKernelAdjoint(xi, xj,resb);
		}
	}



}


mat KernelRegressionModel::calculateKernelMatrix(void) const{

	int Ntraining = trainingData.numberOfSamples;
	mat kernelValuesMatrix = zeros<mat>(Ntraining,Ntraining);


	for(unsigned int i=0; i<Ntraining; i++){

		for(unsigned int j=i; j<Ntraining; j++) {

			rowvec xi = trainingData.getRow(i);
			rowvec xj = trainingData.getRow(j);

			kernelValuesMatrix(i,j) = calculateGaussianKernel(xi, xj);
			kernelValuesMatrix(j,i) = kernelValuesMatrix(i,j);
		}

	}
	return kernelValuesMatrix;


}

double KernelRegressionModel::calculateL1LossAdjoint(mat weights, mat &weightsb) const{

	int Ntraining = trainingData.numberOfSamples;
	double result = 0.0;
	int branch;

	vec yTraining = trainingData.yExact;

	stack<int> stackBranch;
	weightsb.fill(0.0);


	for (unsigned int i = 0; i < Ntraining; ++i) {
		double fSurrogateValue = 0.0;
		for (unsigned int j = 0; j < Ntraining; ++j) {

			if (i != j) {
				fSurrogateValue = fSurrogateValue + yTraining(j)*weights(i,j);
				stackBranch.push(1);
			}
			else {
				stackBranch.push(0);
			}

		}

		double fExact = yTraining(i);
		result += fabs(fExact-fSurrogateValue);
		if (fExact - fSurrogateValue >= 0.0){
			stackBranch.push(0);
		}
		else{
			stackBranch.push(1);
		}

	}

	for (int i = Ntraining-1; i > -1; --i) {

		double fSurrogateValueb = 0.0;
		branch = stackBranch.top(); stackBranch.pop();
		if (branch == 0) {

			fSurrogateValueb = -1.0;
		} else {
			fSurrogateValueb = 1.0;

		}

		for (int j = Ntraining-1; j > -1; --j) {

			branch = stackBranch.top(); stackBranch.pop();
			if (branch != 0) {

				weightsb(i,j) = weightsb(i,j) + yTraining(j)*fSurrogateValueb;
			}
		}
	}
	assert(stackBranch.empty());
	return result;
}


double KernelRegressionModel::calculateL1Loss(mat weights) const{

	int Ntraining = trainingData.numberOfSamples;
	vec yTraining = trainingData.yExact;
	double result = 0.0;

	for(unsigned int i=0; i<Ntraining; i++) {

		double fSurrogateValue = 0.0;
		for(unsigned int j=0; j<Ntraining; j++) {

			if(i!=j) {

				fSurrogateValue += yTraining(j)* weights(i,j);
			}
		}

		double fExact = yTraining(i);
		result += fabs(fExact-fSurrogateValue);

	}
	return result;
}



double KernelRegressionModel::calculateL2LossAdjoint(mat weights, mat &weightsb) const {

	int Ntraining = trainingData.numberOfSamples;
	vec yTraining = trainingData.yExact;
	double result = 0.0;
	int branch;
	stack<double> stackValues;

	weightsb.fill(0.0);

	/* forward sweep */
	for (unsigned int i = 0; i < Ntraining; ++i) {
		double fSurrogateValue = 0.0;
		for (unsigned int j = 0; j < Ntraining; ++j){

			if (i != j) {

				fSurrogateValue = fSurrogateValue + yTraining(j)*weights(i,j);

			}
		}

		double fExact = yTraining(i);

		result += (fExact-fSurrogateValue) * (fExact-fSurrogateValue);
		stackValues.push(fSurrogateValue);

	}

	/* adjoint sweep */


	for (int i = Ntraining-1; i > -1; --i) {

		double fSurrogateValue;
		double fSurrogateValueb = 0.0;


		fSurrogateValue = stackValues.top(); stackValues.pop();

		double fExact = yTraining(i);
		fSurrogateValueb = -(2.0*(fExact-fSurrogateValue));
		for (int j = Ntraining-1; j > -1; --j) {

			if (i != j) {

				weightsb(i,j) = weightsb(i,j) + yTraining(j)*fSurrogateValueb;
			}

		}
	}

	assert(stackValues.empty());
	return result;
}


double KernelRegressionModel::calculateL2Loss(mat weights) const{

	int Ntraining = trainingData.numberOfSamples;
	vec yTraining = trainingData.yExact;

	double result = 0.0;

	for(unsigned int i=0; i<Ntraining; i++) {

		double fSurrogateValue = 0.0;
		for(unsigned int j=0; j<Ntraining; j++) {

			if(i!=j) {

				fSurrogateValue += yTraining(j)* weights(i,j);

			}
		}

		double fExact = yTraining(i);
		result += (fExact-fSurrogateValue) * (fExact-fSurrogateValue);
	}
	return result;
}

double KernelRegressionModel::calculateGaussianKernelAdjoint(rowvec xi, rowvec xj,double calculateGaussianKernelb){

	double twoSigmaSqr = 2.0*sigmaGaussianKernel*sigmaGaussianKernel;
	double twoSigmaSqrb = 0.0;
	double metricVal;
	double metricValb;
	double tempb;
	double temp;
	double tempb0;

	metricVal = calculateMetric(xi, xj, mahalanobisMatrix);

	double kernelVal = 1.0/(sigmaGaussianKernel*rootTwoPi)*exp(-metricVal/twoSigmaSqr);
	kernelVal += EPSILON;

	double kernelValb = calculateGaussianKernelb;
	temp = metricVal/twoSigmaSqr;
	tempb = kernelValb/(rootTwoPi*sigmaGaussianKernel);
	tempb0 = -(exp(-temp)*tempb/twoSigmaSqr);
	metricValb = tempb0;
	twoSigmaSqrb = -(temp*tempb0);
	sigmaGaussianKernelAdjoint += 2.0*sigmaGaussianKernel*2.0*twoSigmaSqrb - exp(-temp)*tempb/sigmaGaussianKernel;
	metricVal = calculateMetricAdjoint(xi, xj, mahalanobisMatrix, mahalanobisMatrixAdjoint, metricValb);
	return kernelVal;
}

double KernelRegressionModel::calculateGaussianKernel(rowvec xi, rowvec xj) const {

	double twoSigmaSqr = 2.0*sigmaGaussianKernel*sigmaGaussianKernel;
	double metricVal = calculateMetric(xi, xj, mahalanobisMatrix);
	double kernelVal = (1.0/(sigmaGaussianKernel*rootTwoPi)) * exp(-metricVal/twoSigmaSqr);
	kernelVal += EPSILON; /* we add some small number to the result for stability */

	assert(kernelVal >= 0);

	return kernelVal;

}




double KernelRegressionModel::calculateL12RegularizationTerm(double weight) const {

	double regTerm = 0.0;

	for (unsigned int i = 0; i < dim; i++)
		for (unsigned int j = 0; j < dim; j++) {

			regTerm += mahalanobisMatrix(i,j) * mahalanobisMatrix(i,j);
		}

	return  weight * regTerm;

}

double KernelRegressionModel::calculateL12RegularizationTermAdjoint(double weight) {


	double regTerm = 0.0;

	for (unsigned int i = 0; i < dim; i++)
		for (unsigned int j = 0; j < dim; j++) {

			regTerm += mahalanobisMatrix(i,j) * mahalanobisMatrix(i,j);
		}

	for (unsigned int i = dim-1; i > -1; --i)
		for (unsigned int j = dim-1; j > -1; --j)
			mahalanobisMatrixAdjoint(i,j) += 2*mahalanobisMatrix(i,j)*weight;

	return  weight * regTerm;
}


double KernelRegressionModel::interpolate(rowvec x) const{
#if 0
	printVector(x,"x");
	printMatrix(X,"X");
#endif
	assert(x.size() == dim);

	unsigned int samplesUsedInInterpolation = X.n_rows;

	vec kernelValues(samplesUsedInInterpolation,fill::zeros);

	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		rowvec xi = X.row(i);
		kernelValues(i) = calculateGaussianKernel(x,xi);

	}

	double kernelSum = sum(kernelValues);

	vec weights(samplesUsedInInterpolation,fill::zeros);

	double weightedSum = 0.0;
	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		weights(i) = kernelValues(i)/kernelSum;
#if 0
		rowvec xi = X.row(i);
		printVector(xi);
		printf("weights(%d) = %10.7f\n",i,weights(i));
#endif
		weightedSum += weights(i)* y(i);
	}


	return weightedSum;


}

void KernelRegressionModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{

	cout <<"ERROR: interpolateWithVariance does not exist for the KernelRegressionModel\n";
	abort();

}


void KernelRegressionModel::setGradientsOn(void){

	this->ifUsesGradientData = true;

}

void KernelRegressionModel::setGradientsOff(void){

	this->ifUsesGradientData = false;

}

double KernelRegressionModel::interpolateWithGradients(rowvec x) const{

	assert(ifUsesGradientData);
	assert(x.size() == dim);

	rowvec xp = normalizeRowVectorBack(x, xmin, xmax);

	unsigned int samplesUsedInInterpolation = X.n_rows;

	vec kernelValues(samplesUsedInInterpolation,fill::zeros);

	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		rowvec xi = X.row(i);
		kernelValues(i) = calculateGaussianKernel(x,xi);

	}

	double kernelSum = sum(kernelValues);

	vec weights(samplesUsedInInterpolation,fill::zeros);


	double weightedSum = 0.0;
	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		rowvec xi = getRowXRaw(i);

		rowvec xdiff = xp-xi;
		rowvec grad = gradientData.row(i);
		double gradTerm = dot(xdiff,grad);

		weights(i) = kernelValues(i)/kernelSum;
		weightedSum += weights(i)* (y(i)+gradTerm);
	}

	return weightedSum;

}



float MAX(float a, float b){

	if(a >=b) return a;
	else return b;


}

//codi::RealReverse MAX(codi::RealReverse a, codi::RealReverse b){
//
//	if(a >=b) return a;
//	else return b;
//
//
//}
//
//
//codi::RealForward MAX(codi::RealForward a, codi::RealForward b){
//
//	if(a >=b) return a;
//	else return b;
//
//
//}


/*
 * calculates the generalized Mahalanobis distance between two points
 *
 * @param[in] x_i : first vector
 * @param[in] X_j : second vector
 * @param[in] M : dim x dim matrix
 * @param[in] dim
 * @return distance
 *
 * */

float calcMetric(float *xi, float *xj, float *M, int dim) {

#if 0
	printf("calling calcMetric (primal)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i*dim+j]);

		}
		printf("\n");
	}

#endif



	float *diff = new float[dim];

	for (int i = 0; i < dim; i++) {

		diff[i] = xi[i] - xj[i];
	}
#if 0
	rowvec xi_val(dim);
	rowvec xj_val(dim);
	rowvec diff_val(dim);
	mat M_val(dim, dim);

	for (int i = 0; i < dim; i++) {
		xi_val(i) = xi[i];
		xj_val(i) = xj[i];
	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			M_val(i, j) = M[i][j];

	diff_val = xi_val - xj_val;

	printf("diff_val=\n");
	diff_val.print();

	colvec diffT = trans(diff_val);

	vec matVecProd = M_val * diffT;
	printf("M * xdiff = \n");
	matVecProd.print();

	float metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	float *tempVec = new float[dim];

	float sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i*dim+j] * diff[j];
		}

		tempVec[i] = sum;
		sum = 0.0;

	}
#if 0
	printf("tempVec = \n");
	for(int i=0; i<dim; i++) {
		printf("%10.7f \n",tempVec[i] );

	}
#endif

	sum = 0.0;

	for (int i = 0; i < dim; i++) {

		sum = sum + tempVec[i] * diff[i];
	}
#if 0
	printf("sum = %10.7f\n",sum);
#endif

	delete[] diff;
	delete[] tempVec;


	if (sum < 0.0) {

		fprintf(stderr, "Error: metric is negative! at FILE = %s, LINE = %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	return sum;

}




double gaussianKernel(rowvec &xi, rowvec &xj, double sigma, mat &M) {
#if 0
	printf("calling gaussianKernel with sigma = %10.7f...\n", sigma);
	xi.print();
	xj.print();
	M.print();
#endif

	/* calculate distance between xi and xj with the matrix M */
	double metricVal = calculateMetric(xi, xj, M);



	double sqr_two_pi = sqrt(2.0 * datum::pi);

	double kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	kernelVal += 10E-14;

#if 0
	printf("metricVal = %15.10f\n",metricVal);
	printf("exp(-metricVal / (2 * sigma * sigma)) = %15.10f\n",exp(-metricVal / (2 * sigma * sigma)));
	printf("kernelVal = %15.10f\n",kernelVal);
#endif



	return kernelVal;

}




double SIGN(double a, double b) {

	if (b >= 0.0) {
		return fabs(a);
	} else {
		return -fabs(a);
	}
}

//codi::RealReverse SIGN(codi::RealReverse a, codi::RealReverse b) {
//
//	if (b >= 0.0) {
//		return fabs(a);
//	} else {
//
//		return -fabs(a);
//	}
//}
//
//codi::RealForward SIGN(codi::RealForward a, codi::RealForward b) {
//
//	if (b >= 0.0) {
//		return fabs(a);
//	} else {
//
//		return -fabs(a);
//	}
//}

double PYTHAG(double a, double b) {
	double at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt) {
		ct = bt / at;
		result = at * sqrt(1.0 + ct * ct);
	} else if (bt > 0.0) {
		ct = at / bt;
		result = bt * sqrt(1.0 + ct * ct);
	} else
		result = 0.0;
	return (result);
}

//codi::RealReverse PYTHAG(codi::RealReverse a, codi::RealReverse b) {
//	codi::RealReverse at = fabs(a), bt = fabs(b), ct, result;
//
//	if (at > bt) {
//		ct = bt / at;
//		result = at * sqrt(1.0 + ct * ct);
//	} else if (bt > 0.0) {
//		ct = at / bt;
//		result = bt * sqrt(1.0 + ct * ct);
//	} else
//		result = 0.0;
//	return (result);
//}
//codi::RealForward PYTHAG(codi::RealForward a, codi::RealForward b) {
//	codi::RealForward at = fabs(a), bt = fabs(b), ct, result;
//
//	if (at > bt) {
//		ct = bt / at;
//		result = at * sqrt(1.0 + ct * ct);
//	} else if (bt > 0.0) {
//		ct = at / bt;
//		result = bt * sqrt(1.0 + ct * ct);
//	} else
//		result = 0.0;
//	return (result);
//}



/** calculate regularization terms for the given matrix L
 *
 * @param[in]  L:  lower diagonal matrix
 * @param[in]  wSvd: weight for the svd regularization part
 * @param[in]  w12: weight for the mixed 12 regularization part
 * @param[out] regTerm
 *
 */

int calcRegTerms(double *L, double *regTerm, double wSvd, double w12, int dim) {
	int flag, i, its, j, jj, k, l = 0, nm;
	double c, f, h, s, x, y, z;
	double anorm = 0.0, g = 0.0, scale = 0.0;


	int m = dim;
	int n = dim;


	double **a;
	a = new double*[dim];

	for (i = 0; i < dim; i++) {

		a[i] = new double[dim];
	}

	double **M;
	M= new double*[dim];

	for (i = 0; i < dim; i++) {

		M[i] = new double[dim];
	}


	double **LT;
	LT = new double*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new double[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i*dim+j];
		}


	}

#if 0
	printf("L = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", L[i*dim+j]);

		}
		printf("\n");
	}
#endif


#if 0
	printf("LT = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", LT[i][j]);

		}
		printf("\n");
	}

#endif

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=0;
			M[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				M[i][j] += L[i*dim+k] * LT[k][j];

			}

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=M[i][j];

		}



#if 0
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j]);

		}
		printf("\n");
	}

#endif

#if 0
	/* only for validation */
	mat Lval(dim,dim);
	mat LTval(dim,dim);
	mat aval(dim,dim);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {
			Lval(i,j) = Lin(i,j);
		}
	LTval = trans(Lval);
	aval = Lval*LTval;

	printf("aval = \n");
	aval.print();

#endif


	/* SVD part */

	double **v;
	v = new double*[n];

	for (i = 0; i < n; i++) {

		v[i] = new double[n];
	}
	double *w = new double[n];

	double *rv1 = new double[n];

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale) {
				for (k = i; k < m; k++) {
					a[k][i] = (a[k][i] / scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1) {
					for (j = l; j < n; j++) {
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i] * scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1) {
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale) {
				for (k = l; k < n; k++) {
					a[i][k] = (a[i][k] / scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1) {
					for (j = l; j < m; j++) {
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k] * scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		if (i < n - 1) {
			if (g) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* float division to avoid underflow */
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++)
						s += (a[i][k] * v[k][j]);
					for (k = l; k < n; k++)
						v[k][j] += (s * v[k][i]);
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g) {
			g = 1.0 / g;
			if (i != n - 1) {
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i] * g);
		} else {
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--) { /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm) {
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) {
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++) {
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k) { /* convergence */
				if (z < 0.0) { /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30000) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return 1;
			}

			/* shift from bottom 2 x 2 minor */
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) {
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) {
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = (y * c + z * s);
					a[jj][i] = (z * c - y * s);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	delete[] rv1;

#if 0
	printf("singular values of a=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i]);
	}
#endif

	/* sort the singular values */

	double temp;
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {

			if (w[i] < w[j])

			{
				temp = w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 0
	printf("singular values of a=\n");


	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i]);
	}
#endif

	/* normalization */
	double wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}

#if 0
	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum);


	for (i = 0; i < n; i++) {

		printf("%15.10f\n",w[i]);
	}
#endif


	double svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;
#if 0
	printf("svd_multiplier = %10.7f\n",svd_multiplier);
#endif
	float reg_term_svd = 0.0;

	for (i = 0; i < n; i++) {
#if 0
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		reg_term_svd = reg_term_svd + (i + 1) * w[i];
	}
#if 0
	printf("reg_term_svd = %10.7f\n",reg_term_svd);
#endif


	double reg_term_L1 = 0.0;

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {

			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1);
#endif



	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] M[i];
		delete[] LT[i];
	}


	delete[] LT;
	delete[] M;
	delete[] a;
	delete[] v;
	delete[] w;



	*regTerm = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
#if 0
	printf("result = %10.7f\n",*regTerm);
#endif

	return 0;




}

///* forward mode */
//
//int calcRegTerms(double *L, double *regTerm, double *regTermd, double wSvd, double w12, int dim, int derIndx) {
//	int flag, i, its, j, jj, k, l = 0, nm;
//	codi::RealForward c, f, h, s, x, y, z;
//	codi::RealForward anorm = 0.0, g = 0.0, scale = 0.0;
//
//
//	int m = dim;
//	int n = dim;
//
//	codi::RealForward *Lcodi = new codi::RealForward[dim*dim];
//	for (int i = 0; i < dim*dim; i++) {
//
//		Lcodi[i] = L[i];
//
//	}
//
//	Lcodi[derIndx].setGradient(1.0);
//
//
//	codi::RealForward **a;
//	a = new codi::RealForward*[dim];
//
//	for (i = 0; i < dim; i++) {
//
//		a[i] = new codi::RealForward[dim];
//	}
//
//	codi::RealForward **M;
//	M= new codi::RealForward*[dim];
//
//	for (i = 0; i < dim; i++) {
//
//		M[i] = new codi::RealForward[dim];
//	}
//
//
//	codi::RealForward **LT;
//	LT = new codi::RealForward*[dim];
//	for (int i = 0; i < dim; i++) {
//		LT[i] = new codi::RealForward[dim];
//
//	}
//
//	for (int i = 0; i < dim; i++)
//		for (int j = 0; j < dim; j++) {
//
//			LT[i][j]=0.0;
//		}
//
//
//
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j <= i; j++){
//
//			LT[j][i] = Lcodi[i*dim+j];
//		}
//
//
//	}
//#if 0
//	printf("LT = \n");
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j < dim; j++) {
//
//			printf("%10.7f ", LT[i][j]);
//
//		}
//		printf("\n");
//	}
//
//#endif
//
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//		{
//			a[i][j]=0;
//			M[i][j]=0;
//		}
//
//	/* Multiplying matrix L and LT and storing in M */
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//			for(int k = 0; k < dim; ++k)
//			{
//				M[i][j] += Lcodi[i*dim+k] * LT[k][j];
//
//			}
//
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//		{
//			a[i][j]=M[i][j];
//
//		}
//
//
//
//#if 0
//	printf("a = \n");
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j < dim; j++) {
//
//			printf("%10.7f ", a[i][j]);
//
//		}
//		printf("\n");
//	}
//
//#endif
//
//
//
//
//	/* SVD part */
//
//	codi::RealForward **v;
//	v = new codi::RealForward*[n];
//
//	for (i = 0; i < n; i++) {
//
//		v[i] = new codi::RealForward[n];
//	}
//	codi::RealForward *w = new codi::RealForward[n];
//
//	codi::RealForward *rv1 = new codi::RealForward[n];
//
//	/* Householder reduction to bidiagonal form */
//	for (i = 0; i < n; i++) {
//		/* left-hand reduction */
//		l = i + 1;
//		rv1[i] = scale * g;
//		g = s = scale = 0.0;
//		if (i < m) {
//			for (k = i; k < m; k++)
//				scale += fabs(a[k][i]);
//			if (scale!= 0) {
//				for (k = i; k < m; k++) {
//					a[k][i] = (a[k][i] / scale);
//					s += (a[k][i] * a[k][i]);
//				}
//				f = a[i][i];
//				g = -SIGN(sqrt(s), f);
//				h = f * g - s;
//				a[i][i] = (f - g);
//				if (i != n - 1) {
//					for (j = l; j < n; j++) {
//						for (s = 0.0, k = i; k < m; k++)
//							s += (a[k][i] * a[k][j]);
//						f = s / h;
//						for (k = i; k < m; k++)
//							a[k][j] += (f * a[k][i]);
//					}
//				}
//				for (k = i; k < m; k++)
//					a[k][i] = (a[k][i] * scale);
//			}
//		}
//		w[i] = (scale * g);
//
//		/* right-hand reduction */
//		g = s = scale = 0.0;
//		if (i < m && i != n - 1) {
//			for (k = l; k < n; k++)
//				scale += fabs(a[i][k]);
//			if (scale!=0) {
//				for (k = l; k < n; k++) {
//					a[i][k] = (a[i][k] / scale);
//					s += (a[i][k] * a[i][k]);
//				}
//				f = a[i][l];
//				g = -SIGN(sqrt(s), f);
//				h = f * g - s;
//				a[i][l] = (f - g);
//				for (k = l; k < n; k++)
//					rv1[k] = a[i][k] / h;
//				if (i != m - 1) {
//					for (j = l; j < m; j++) {
//						for (s = 0.0, k = l; k < n; k++)
//							s += (a[j][k] * a[i][k]);
//						for (k = l; k < n; k++)
//							a[j][k] += (s * rv1[k]);
//					}
//				}
//				for (k = l; k < n; k++)
//					a[i][k] = (a[i][k] * scale);
//			}
//		}
//		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
//	}
//
//	/* accumulate the right-hand transformation */
//	for (i = n - 1; i >= 0; i--) {
//		if (i < n - 1) {
//			if (g!=0) {
//				for (j = l; j < n; j++)
//					v[j][i] = ((a[i][j] / a[i][l]) / g);
//				/* float division to avoid underflow */
//				for (j = l; j < n; j++) {
//					for (s = 0.0, k = l; k < n; k++)
//						s += (a[i][k] * v[k][j]);
//					for (k = l; k < n; k++)
//						v[k][j] += (s * v[k][i]);
//				}
//			}
//			for (j = l; j < n; j++)
//				v[i][j] = v[j][i] = 0.0;
//		}
//		v[i][i] = 1.0;
//		g = rv1[i];
//		l = i;
//	}
//
//	/* accumulate the left-hand transformation */
//	for (i = n - 1; i >= 0; i--) {
//		l = i + 1;
//		g = w[i];
//		if (i < n - 1)
//			for (j = l; j < n; j++)
//				a[i][j] = 0.0;
//		if (g!=0) {
//			g = 1.0 / g;
//			if (i != n - 1) {
//				for (j = l; j < n; j++) {
//					for (s = 0.0, k = l; k < m; k++)
//						s += (a[k][i] * a[k][j]);
//					f = (s / a[i][i]) * g;
//					for (k = i; k < m; k++)
//						a[k][j] += (f * a[k][i]);
//				}
//			}
//			for (j = i; j < m; j++)
//				a[j][i] = (a[j][i] * g);
//		} else {
//			for (j = i; j < m; j++)
//				a[j][i] = 0.0;
//		}
//		++a[i][i];
//	}
//
//	/* diagonalize the bidiagonal form */
//	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
//		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
//			flag = 1;
//			for (l = k; l >= 0; l--) { /* test for splitting */
//				nm = l - 1;
//				if (fabs(rv1[l]) + anorm == anorm) {
//					flag = 0;
//					break;
//				}
//				if (fabs(w[nm]) + anorm == anorm)
//					break;
//			}
//			if (flag) {
//				c = 0.0;
//				s = 1.0;
//				for (i = l; i <= k; i++) {
//					f = s * rv1[i];
//					if (fabs(f) + anorm != anorm) {
//						g = w[i];
//						h = PYTHAG(f, g);
//						w[i] = h;
//						h = 1.0 / h;
//						c = g * h;
//						s = (-f * h);
//						for (j = 0; j < m; j++) {
//							y = a[j][nm];
//							z = a[j][i];
//							a[j][nm] = (y * c + z * s);
//							a[j][i] = (z * c - y * s);
//						}
//					}
//				}
//			}
//			z = w[k];
//			if (l == k) { /* convergence */
//				if (z < 0.0) { /* make singular value nonnegative */
//					w[k] = (-z);
//					for (j = 0; j < n; j++)
//						v[j][k] = (-v[j][k]);
//				}
//				break;
//			}
//			if (its >= 30000) {
//				delete[] rv1;
//				fprintf(stderr, "No convergence after 30,000! iterations \n");
//				return 1;
//			}
//
//			/* shift from bottom 2 x 2 minor */
//			x = w[l];
//			nm = k - 1;
//			y = w[nm];
//			g = rv1[nm];
//			h = rv1[k];
//			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
//			g = PYTHAG(f, 1.0);
//			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
//
//			/* next QR transformation */
//			c = s = 1.0;
//			for (j = l; j <= nm; j++) {
//				i = j + 1;
//				g = rv1[i];
//				y = w[i];
//				h = s * g;
//				g = c * g;
//				z = PYTHAG(f, h);
//				rv1[j] = z;
//				c = f / z;
//				s = h / z;
//				f = x * c + g * s;
//				g = g * c - x * s;
//				h = y * s;
//				y = y * c;
//				for (jj = 0; jj < n; jj++) {
//					x = v[jj][j];
//					z = v[jj][i];
//					v[jj][j] = (x * c + z * s);
//					v[jj][i] = (z * c - x * s);
//				}
//				z = PYTHAG(f, h);
//				w[j] = z;
//				if (z!=0) {
//					z = 1.0 / z;
//					c = f * z;
//					s = h * z;
//				}
//				f = (c * g) + (s * y);
//				x = (c * y) - (s * g);
//				for (jj = 0; jj < m; jj++) {
//					y = a[jj][j];
//					z = a[jj][i];
//					a[jj][j] = (y * c + z * s);
//					a[jj][i] = (z * c - y * s);
//				}
//			}
//			rv1[l] = 0.0;
//			rv1[k] = f;
//			w[k] = x;
//		}
//	}
//	delete[] rv1;
//
//#if 0
//	printf("singular values of a=\n");
//
//	for (i = 0; i < n; i++) {
//
//		printf("%10.7f\n",w[i]);
//	}
//#endif
//
//	/* sort the singular values */
//
//	codi::RealForward temp;
//	for (i = 0; i < n; ++i) {
//		for (j = i + 1; j < n; ++j) {
//
//			if (w[i] < w[j])
//
//			{
//				temp = w[i];
//				w[i] = w[j];
//				w[j] = temp;
//			}
//		}
//	}
//
//#if 0
//	printf("singular values of a=\n");
//
//
//	for (i = 0; i < n; i++) {
//
//		printf("%10.7f\n",w[i].getValue());
//	}
//#endif
//
//	/* normalization */
//	codi::RealForward wsum = 0.0;
//	for (i = 0; i < n; i++) {
//
//		wsum += w[i];
//
//	}
//
//	for (i = 0; i < n; i++) {
//
//		w[i] = w[i]/wsum;
//
//	}
//
//#if 0
//	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum.getValue());
//
//
//	for (i = 0; i < n; i++) {
//
//		printf("%15.10f\n",w[i].getValue());
//	}
//#endif
//
//
//	float svd_multiplier = (1.0*n*(1.0*n+1))/2.0;
//
//	svd_multiplier = 1.0/svd_multiplier;
//#if 0
//	printf("svd_multiplier = %10.7f\n",svd_multiplier);
//#endif
//	codi::RealForward reg_term_svd = 0.0;
//
//	for (i = 0; i < n; i++) {
//#if 0
//		printf("%d * %10.7f = %10.7f\n",i+1,w[i].getValue(),(i+1)*w[i].getValue());
//#endif
//		reg_term_svd = reg_term_svd + (i + 1) * w[i];
//	}
//#if 0
//	printf("reg_term_svd = %10.7f\n",reg_term_svd.getValue());
//#endif
//
//
//	codi::RealForward reg_term_L1 = 0.0;
//
//	for (i = 0; i < n; i++)
//		for (j = 0; j < n; j++) {
//
//			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
//		}
//#if 0
//	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
//#endif
//
//
//
//	for (i = 0; i < n; i++) {
//		delete[] v[i];
//		delete[] a[i];
//		delete[] M[i];
//		delete[] LT[i];
//	}
//
//
//	delete[] LT;
//	delete[] M;
//	delete[] a;
//	delete[] v;
//	delete[] w;
//
//
//	codi::RealForward result = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
//	*regTerm = result.getValue();
//
//	*regTermd = result.getGradient();
//
//	return 0;
//
//
//}



//int calcRegTerms(double *L, double *Lb, double *result, double wSvd, double w12, int dim) {
//	int flag, i, its, j, jj, k, l = 0, nm;
//
//
//	codi::RealReverse *Lcodi = new codi::RealReverse[dim*dim];
//	for (int i = 0; i < dim*dim; i++) {
//
//		Lcodi[i] = L[i];
//
//	}
//
//	/* activate tape and register input */
//
//	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
//	tape.setActive();
//
//	codi::RealReverse regTerm=0.0;
//
//
//
//	for (int i = 0; i < dim*dim; i++) {
//
//		tape.registerInput(Lcodi[i]);
//
//	}
//
//
//	codi::RealReverse c, f, h, s, x, y, z;
//	codi::RealReverse anorm = 0.0, g = 0.0, scale = 0.0;
//
//
//	int m = dim;
//	int n = dim;
//
//
//	codi::RealReverse **a;
//	a = new codi::RealReverse*[dim];
//
//	for (i = 0; i < dim; i++) {
//
//		a[i] = new codi::RealReverse[dim];
//	}
//
//	codi::RealReverse **M;
//	M = new codi::RealReverse*[dim];
//
//	for (i = 0; i < dim; i++) {
//
//		M[i] = new codi::RealReverse[dim];
//	}
//
//
//
//
//	codi::RealReverse **LT;
//	LT = new codi::RealReverse*[dim];
//	for (int i = 0; i < dim; i++) {
//		LT[i] = new codi::RealReverse[dim];
//
//	}
//
//	for (int i = 0; i < dim; i++)
//		for (int j = 0; j < dim; j++) {
//
//			LT[i][j]=0.0;
//		}
//
//
//
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j <= i; j++){
//
//			LT[j][i] = Lcodi[i*dim+j];
//		}
//
//
//	}
//#if 0
//	printf("LT = \n");
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j < dim; j++) {
//
//			printf("%10.7f ", LT[i][j].getValue());
//
//		}
//		printf("\n");
//	}
//
//#endif
//
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//		{
//			a[i][j]=0;
//			M[i][j]=0;
//		}
//
//	/* Multiplying matrix L and LT and storing in M */
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//			for(int k = 0; k < dim; ++k)
//			{
//				M[i][j] += Lcodi[i*dim+k] * LT[k][j];
//
//			}
//
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//		{
//			a[i][j]=M[i][j];
//
//		}
//
//#if 0
//	printf("M = \n");
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j < dim; j++) {
//
//			printf("%10.7f ", M[i][j].getValue());
//
//		}
//		printf("\n");
//	}
//
//#endif
//
//#if 0
//	printf("a = \n");
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j < dim; j++) {
//
//			printf("%10.7f ", a[i][j].getValue());
//
//		}
//		printf("\n");
//	}
//
//#endif
//
//
//	/* SVD part */
//
//	codi::RealReverse **v;
//	v = new codi::RealReverse*[n];
//
//	for (i = 0; i < n; i++) {
//
//		v[i] = new codi::RealReverse[n];
//	}
//	codi::RealReverse *w = new codi::RealReverse[n];
//
//	codi::RealReverse *rv1 = new codi::RealReverse[n];
//
//	/* Householder reduction to bidiagonal form */
//	for (i = 0; i < n; i++) {
//		/* left-hand reduction */
//		l = i + 1;
//		rv1[i] = scale * g;
//		g = s = scale = 0.0;
//		if (i < m) {
//			for (k = i; k < m; k++)
//				scale += fabs(a[k][i]);
//			if (scale != 0) {
//				for (k = i; k < m; k++) {
//					a[k][i] = (a[k][i] / scale);
//					s += (a[k][i] * a[k][i]);
//				}
//				f = a[i][i];
//				g = -SIGN(sqrt(s), f);
//				h = f * g - s;
//				a[i][i] = (f - g);
//				if (i != n - 1) {
//					for (j = l; j < n; j++) {
//						for (s = 0.0, k = i; k < m; k++)
//							s += (a[k][i] * a[k][j]);
//						f = s / h;
//						for (k = i; k < m; k++)
//							a[k][j] += (f * a[k][i]);
//					}
//				}
//				for (k = i; k < m; k++)
//					a[k][i] = (a[k][i] * scale);
//			}
//		}
//		w[i] = (scale * g);
//
//		/* right-hand reduction */
//		g = s = scale = 0.0;
//		if (i < m && i != n - 1) {
//			for (k = l; k < n; k++)
//				scale += fabs(a[i][k]);
//			if (scale !=0) {
//				for (k = l; k < n; k++) {
//					a[i][k] = (a[i][k] / scale);
//					s += (a[i][k] * a[i][k]);
//				}
//				f = a[i][l];
//				g = -SIGN(sqrt(s), f);
//				h = f * g - s;
//				a[i][l] = (f - g);
//				for (k = l; k < n; k++)
//					rv1[k] = a[i][k] / h;
//				if (i != m - 1) {
//					for (j = l; j < m; j++) {
//						for (s = 0.0, k = l; k < n; k++)
//							s += (a[j][k] * a[i][k]);
//						for (k = l; k < n; k++)
//							a[j][k] += (s * rv1[k]);
//					}
//				}
//				for (k = l; k < n; k++)
//					a[i][k] = (a[i][k] * scale);
//			}
//		}
//		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
//	}
//
//	/* accumulate the right-hand transformation */
//	for (i = n - 1; i >= 0; i--) {
//		if (i < n - 1) {
//			if (g !=0) {
//				for (j = l; j < n; j++)
//					v[j][i] = ((a[i][j] / a[i][l]) / g);
//				/* float division to avoid underflow */
//				for (j = l; j < n; j++) {
//					for (s = 0.0, k = l; k < n; k++)
//						s += (a[i][k] * v[k][j]);
//					for (k = l; k < n; k++)
//						v[k][j] += (s * v[k][i]);
//				}
//			}
//			for (j = l; j < n; j++)
//				v[i][j] = v[j][i] = 0.0;
//		}
//		v[i][i] = 1.0;
//		g = rv1[i];
//		l = i;
//	}
//
//	/* accumulate the left-hand transformation */
//	for (i = n - 1; i >= 0; i--) {
//		l = i + 1;
//		g = w[i];
//		if (i < n - 1)
//			for (j = l; j < n; j++)
//				a[i][j] = 0.0;
//		if (g != 0) {
//			g = 1.0 / g;
//			if (i != n - 1) {
//				for (j = l; j < n; j++) {
//					for (s = 0.0, k = l; k < m; k++)
//						s += (a[k][i] * a[k][j]);
//					f = (s / a[i][i]) * g;
//					for (k = i; k < m; k++)
//						a[k][j] += (f * a[k][i]);
//				}
//			}
//			for (j = i; j < m; j++)
//				a[j][i] = (a[j][i] * g);
//		} else {
//			for (j = i; j < m; j++)
//				a[j][i] = 0.0;
//		}
//		++a[i][i];
//	}
//
//	/* diagonalize the bidiagonal form */
//	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
//		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
//			flag = 1;
//			for (l = k; l >= 0; l--) { /* test for splitting */
//				nm = l - 1;
//				if (fabs(rv1[l]) + anorm == anorm) {
//					flag = 0;
//					break;
//				}
//				if (fabs(w[nm]) + anorm == anorm)
//					break;
//			}
//			if (flag) {
//				c = 0.0;
//				s = 1.0;
//				for (i = l; i <= k; i++) {
//					f = s * rv1[i];
//					if (fabs(f) + anorm != anorm) {
//						g = w[i];
//						h = PYTHAG(f, g);
//						w[i] = h;
//						h = 1.0 / h;
//						c = g * h;
//						s = (-f * h);
//						for (j = 0; j < m; j++) {
//							y = a[j][nm];
//							z = a[j][i];
//							a[j][nm] = (y * c + z * s);
//							a[j][i] = (z * c - y * s);
//						}
//					}
//				}
//			}
//			z = w[k];
//			if (l == k) { /* convergence */
//				if (z < 0.0) { /* make singular value nonnegative */
//					w[k] = (-z);
//					for (j = 0; j < n; j++)
//						v[j][k] = (-v[j][k]);
//				}
//				break;
//			}
//			if (its >= 30000) {
//				delete[] rv1;
//				fprintf(stderr, "No convergence after 30,000! iterations \n");
//				return 1;
//			}
//
//			/* shift from bottom 2 x 2 minor */
//			x = w[l];
//			nm = k - 1;
//			y = w[nm];
//			g = rv1[nm];
//			h = rv1[k];
//			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
//			g = PYTHAG(f, 1.0);
//			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
//
//			/* next QR transformation */
//			c = s = 1.0;
//			for (j = l; j <= nm; j++) {
//				i = j + 1;
//				g = rv1[i];
//				y = w[i];
//				h = s * g;
//				g = c * g;
//				z = PYTHAG(f, h);
//				rv1[j] = z;
//				c = f / z;
//				s = h / z;
//				f = x * c + g * s;
//				g = g * c - x * s;
//				h = y * s;
//				y = y * c;
//				for (jj = 0; jj < n; jj++) {
//					x = v[jj][j];
//					z = v[jj][i];
//					v[jj][j] = (x * c + z * s);
//					v[jj][i] = (z * c - x * s);
//				}
//				z = PYTHAG(f, h);
//				w[j] = z;
//				if (z != 0) {
//					z = 1.0 / z;
//					c = f * z;
//					s = h * z;
//				}
//				f = (c * g) + (s * y);
//				x = (c * y) - (s * g);
//				for (jj = 0; jj < m; jj++) {
//					y = a[jj][j];
//					z = a[jj][i];
//					a[jj][j] = (y * c + z * s);
//					a[jj][i] = (z * c - y * s);
//				}
//			}
//			rv1[l] = 0.0;
//			rv1[k] = f;
//			w[k] = x;
//		}
//	}
//	delete[] rv1;
//
//#if 0
//	printf("singular values of a=\n");
//
//	for (i = 0; i < n; i++) {
//
//		printf("%10.7f\n",w[i]);
//	}
//#endif
//
//	codi::RealReverse temp;
//	for (i = 0; i < n; ++i) {
//		for (j = i + 1; j < n; ++j) {
//
//			if (w[i] < w[j])
//
//			{
//				temp = w[i];
//				w[i] = w[j];
//				w[j] = temp;
//			}
//		}
//	}
//
//#if 0
//	printf("singular values of a=\n");
//
//
//	for (i = 0; i < n; i++) {
//
//		printf("%10.7f\n",w[i].getValue());
//	}
//#endif
//
//	codi::RealReverse wsum = 0.0;
//	for (i = 0; i < n; i++) {
//
//		wsum += w[i];
//
//	}
//
//	for (i = 0; i < n; i++) {
//
//		w[i] = w[i]/wsum;
//
//	}
//
//#if 0
//	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum.getValue());
//
//
//	for (i = 0; i < n; i++) {
//
//		printf("%15.10f\n",w[i].getValue());
//	}
//#endif
//
//
//	codi::RealReverse svd_multiplier = (1.0*n*(1.0*n+1))/2.0;
//
//	svd_multiplier = 1.0/svd_multiplier;
//#if 0
//	printf("svd_multiplier = %10.7f\n",svd_multiplier);
//#endif
//	codi::RealReverse reg_term_svd = 0.0;
//
//	for (i = 0; i < n; i++) {
//#if 0
//		printf("%d * %10.7f = %10.7f\n",i+1,w[i].getValue(),(i+1)*w[i].getValue());
//#endif
//		reg_term_svd = reg_term_svd + (i + 1) * w[i];
//	}
//#if 0
//	printf("reg_term_svd = %10.7f\n",reg_term_svd.getValue());
//#endif
//
//
//	codi::RealReverse reg_term_L1 = 0.0;
//
//	for (i = 0; i < n; i++)
//		for (j = 0; j < n; j++) {
//
//			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
//		}
//#if 0
//	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
//#endif
//
//
//
//
//
//
//
//	regTerm = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
//
//#if 0
//	printf("w12 * reg_term_L1 = %10.7f\n",w12 * reg_term_L1.getValue());
//#endif
//
//
//	tape.registerOutput(regTerm);
//
//	tape.setPassive();
//	regTerm.setGradient(1.0);
//	tape.evaluate();
//
//	for (int i = 0; i < dim*dim; i++) {
//
//
//		Lb[i] = Lcodi[i].getGradient();
//
//	}
//
//
//	tape.reset();
//
//	*result = regTerm.getValue();
//
//	for (i = 0; i < n; i++) {
//
//		delete[] v[i];
//		delete[] a[i];
//		delete[] M[i];
//		delete[] LT[i];
//	}
//
//
//	delete[] LT;
//	delete[] M;
//	delete[] a;
//	delete[] v;
//	delete[] w;
//	delete[] Lcodi;
//
//
//	return 0;
//
//
//
//
//}


/** calculate regularization terms for the given matrix L
 *
 * @param[in]  L:  lower diagonal matrix
 * @param[in]  w12: weight for the mixed 12 regularization part
 * @param[out] regTerm
 *
 */

int calcRegTermL12(double *L, double *regTerm, double w12, int dim) {


	double **M;
	M= new double*[dim];

	for (int i = 0; i < dim; i++) {

		M[i] = new double[dim];
	}


	double **LT;
	LT = new double*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new double[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i*dim+j];
		}


	}



	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{

			M[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				M[i][j] += L[i*dim+k] * LT[k][j];

			}


	double reg_term_L1 = 0.0;

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1);
#endif



	for (int i = 0; i < dim; i++) {
		delete[] M[i];
		delete[] LT[i];
	}


	delete[] LT;
	delete[] M;


	*regTerm = w12 * reg_term_L1;
#if 0
	printf("result = %10.7f\n",*regTerm);
#endif

	return 0;




}


//int calcRegTermL12(double *L, double *Lb, double *result , double w12, int dim) {
//
//
//
//	codi::RealReverse *Lcodi = new codi::RealReverse[dim*dim];
//	for (int i = 0; i < dim*dim; i++) {
//
//		Lcodi[i] = L[i];
//
//	}
//
//	/* activate tape and register input */
//
//	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
//	tape.setActive();
//
//	codi::RealReverse regTerm=0.0;
//
//
//
//	for (int i = 0; i < dim*dim; i++) {
//
//		tape.registerInput(Lcodi[i]);
//
//	}
//
//
//
//
//
//	codi::RealReverse **M;
//	M = new codi::RealReverse*[dim];
//
//	for (int i = 0; i < dim; i++) {
//
//		M[i] = new codi::RealReverse[dim];
//	}
//
//
//
//
//	codi::RealReverse **LT;
//	LT = new codi::RealReverse*[dim];
//	for (int i = 0; i < dim; i++) {
//		LT[i] = new codi::RealReverse[dim];
//
//	}
//
//	for (int i = 0; i < dim; i++)
//		for (int j = 0; j < dim; j++) {
//
//			LT[i][j]=0.0;
//		}
//
//
//
//
//	for (int i = 0; i < dim; i++) {
//		for (int j = 0; j <= i; j++){
//
//			LT[j][i] = Lcodi[i*dim+j];
//		}
//
//
//	}
//
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//		{
//
//			M[i][j]=0;
//		}
//
//	/* Multiplying matrix L and LT and storing in M */
//	for(int i = 0; i < dim; ++i)
//		for(int j = 0; j < dim; ++j)
//			for(int k = 0; k < dim; ++k)
//			{
//				M[i][j] += Lcodi[i*dim+k] * LT[k][j];
//
//			}
//
//
//
//
//
//
//
//	codi::RealReverse reg_term_L1 = 0.0;
//
//	for (int i = 0; i < dim; i++)
//		for (int j = 0; j < dim; j++) {
//
//			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
//		}
//
//
//	regTerm = w12 * reg_term_L1;
//
//#if 0
//	printf("w12 * reg_term_L1 = %10.7f\n",w12 * reg_term_L1.getValue());
//#endif
//
//
//	tape.registerOutput(regTerm);
//
//	tape.setPassive();
//	regTerm.setGradient(1.0);
//	tape.evaluate();
//
//	for (int i = 0; i < dim*dim; i++) {
//
//
//		Lb[i] = Lcodi[i].getGradient();
//
//	}
//
//
//	tape.reset();
//
//	*result = regTerm.getValue();
//
//	for (int i = 0; i < dim; i++) {
//
//		delete[] M[i];
//		delete[] LT[i];
//	}
//
//
//	delete[] LT;
//	delete[] M;
//	delete[] Lcodi;
//
//
//	return 0;
//
//
//
//
//}



void calculateKernelValues(double *X, double *kernelValTable, double *M, double sigma, int N, int d){
#if 0
	printf("calculateKernelValues...\n");
#endif

	const double sqr_two_pi = sqrt(2.0 * 3.14159265359);

	double diff[d];
	double tempVec[d];

	for(int indx1=0; indx1<N; indx1++){
		for(int indx2=indx1+1; indx2<N; indx2++){

#if 0
			printf("indx1 = %d, indx2 = %d\n",indx1,indx2);
#endif

			int off1 = indx1*(d+1);
			int off2 = indx2*(d+1);



			for (int k = 0; k < d; k++) {

				diff[k] = X[off1+k] - X[off2+k];
#if 0
				printf("diff[%d] = %10.7f\n",k,diff[k]);
#endif


			}

			double sum = 0.0;

			for (int i = 0; i < d; i++) {
				for (int j = 0; j < d; j++) {

					sum = sum + M[i*d+j] * diff[j];
				}

				tempVec[i] = sum;
				sum = 0.0;

			}
			sum = 0.0;

			for (int i = 0; i < d; i++) {

				sum = sum + tempVec[i] * diff[i];
			}

			double kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-sum / (2 * sigma * sigma)) + 10E-12;

			kernelValTable[indx1*N+indx2]= kernelVal;


		}
	}



}


void calculateKernelValues_b(double *X, double *kernelValTable, double *kernelValTableb, double *M, double *Mb, double sigma, double *sigmab, int N, int d) {

	const double sqr_two_pi = sqrt(2.0*3.14159265359);
	double diff[d];
	//	double diffb[d];
	double tempVec[d];
	double tempVecb[d];


	//	stack<float> stack;

	*sigmab=0.0;
	for (int i = 0; i < d*d; i++) {

		Mb[i]=0.0;
	}


	int ii1;
	for (int indx1 = 0; indx1 < N; ++indx1) {

		for (int indx2 = indx1+1; indx2 < N; ++indx2) {
			int off1 = indx1*(d+1);
			int off2 = indx2*(d+1);
			for (int k = 0; k < d; ++k) {
				//				stack.push(diff[k]);
				diff[k] = X[off1 + k] - X[off2 + k];
			}
			double sum = 0.0;
			for (int i = 0; i < d; ++i) {
				for (int j = 0; j < d; ++j)
					sum = sum + M[i*d+j]*diff[j];
				tempVec[i] = sum;
				sum = 0.0;
			}
			sum = 0.0;
			for (int i = 0; i < d; ++i) {
				sum = sum + tempVec[i]*diff[i];
			}

			double temp = 2*(sigma*sigma);
			double temp0 = sum/temp;
			double temp1 = exp(-temp0);
			//			double kernelVal = 1.0/(sigma*sqr_two_pi)*temp1 + 10E-12;
			//			stack.push(sum);
			//		}

			//	}

			for (ii1 = 0; ii1 < d; ++ii1)
				tempVecb[ii1] = 0.0;


			//	for (int indx1 = N-1; indx1 > -1; --indx1) {

			//		for (int indx2 = N-1; indx2 > indx1; --indx2) {
			//			int off1;
			//			int off2;
			//			float sum;
			double sumb = 0.0;
			//			float kernelVal;
			double kernelValb = 0.0;
			double tempb;

			double tempb0;
			//			sum = stack.top(); stack.pop();
			//             popReal4(&sum);
			kernelValb = kernelValTableb[indx1*N + indx2];
			kernelValTableb[indx1*N + indx2] = 0.0;
			tempb = kernelValb/(sqr_two_pi*sigma);

			tempb0 = -(exp(-temp0)*tempb/temp);
			sumb = tempb0;
			*sigmab = *sigmab - temp1*tempb/sigma - 2*2*temp0*sigma*tempb0;


			for (int i = d-1; i > -1; --i)
				tempVecb[i] = tempVecb[i] + diff[i]*sumb;
			for (int i = d-1; i > -1; --i) {
				sumb = tempVecb[i];
				tempVecb[i] = 0.0;
				for (int j = d-1; j > -1; --j)
					Mb[i*d + j] = Mb[i*d + j] + diff[j]*sumb;
			}
			//			for (int k = d-1; k > -1; --k) {
			//				diff[k] = stack.top();
			//				stack.pop();
			//			}
			//                 popReal4(&(diff[k]));
		}
	}

}


//void calculateKernelValues(double *X, codi::RealReverse *kernelValTable, codi::RealReverse *M, codi::RealReverse sigma, int N, int d){
//
//	const double sqr_two_pi = sqrt(2.0 * 3.14159265359);
//
//	codi::RealReverse diff[d];
//	codi::RealReverse tempVec[d];
//
//	for(int indx1=0; indx1<N; indx1++){
//		for(int indx2=indx1+1; indx2<N; indx2++){
//
//
//
//			int off1 = indx1*(d+1);
//			int off2 = indx2*(d+1);
//
//
//
//			for (int k = 0; k < d; k++) {
//
//				diff[k] = X[off1+k] - X[off2+k];
//
//
//			}
//
//
//
//
//
//			codi::RealReverse sum = 0.0;
//
//			for (int i = 0; i < d; i++) {
//				for (int j = 0; j < d; j++) {
//
//					sum = sum + M[i*d+j] * diff[j];
//				}
//
//				tempVec[i] = sum;
//				sum = 0.0;
//
//			}
//
//
//			sum = 0.0;
//
//			for (int i = 0; i < d; i++) {
//
//				sum = sum + tempVec[i] * diff[i];
//			}
//
//
//
//
//			codi::RealReverse kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-sum / (2 * sigma * sigma)) + 10E-12;
//
//			kernelValTable[indx1*N+indx2]= kernelVal;
//
//
//		}
//	}
//
//
//
//}

void calculateLossKernelL1(double *result, double *data, double *kernelValTable, int N, int d){


	double lossFunc = 0.0;

	for(int tid=0; tid<N; tid++){



		double kernelSum = 0.0;

		for(int i=0; i<N; i++){

			if(tid != i){

				int indxKernelValTable;
				if(i<tid) {


					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				kernelSum += kernelValTable[indxKernelValTable];

			}



		}

		double fapprox=0.0;
		for(int i=0; i<N; i++){

			if(tid != i){
				int indxKernelValTable;

				if(i<tid) {

					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				fapprox += (kernelValTable[indxKernelValTable]/kernelSum)* data[i*(d+1)+d];

			}




		}

		lossFunc += fabs(fapprox - data[tid*(d+1)+d]);

	}

	*result = lossFunc;

}


void calculateLossKernelL1_b(double *result, double *resultb, double *data, double
		*kernelValTable, double *kernelValTableb, int N, int d) {
	double lossFunc = 0.0;
	double lossFuncb = 0.0;
	int branch;

	stack<int> stackInt;
	stack<int> stackCont;
	stack<double> stackReal;


	for(int i=0; i<N*N; i++) {
		kernelValTableb[i]=0.0;
	}

	for (int tid = 0; tid < N; ++tid) {
		double kernelSum = 0.0;
		//		double fabs0;
		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				kernelSum = kernelSum + kernelValTable[indxKernelValTable];
				stackInt.push(indxKernelValTable);
				//				pushInteger4(indxKernelValTable);
				stackCont.push(1);
				//				pushControl1b(1);
			} else
				stackCont.push(0);
			//				pushControl1b(0);

		}
		double fapprox = 0.0;
		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				fapprox = fapprox + kernelValTable[indxKernelValTable]/
						kernelSum*data[i*(d+1)+d];
				stackInt.push(indxKernelValTable);
				//				pushInteger4(indxKernelValTable);
				//				pushControl1b(1);
				stackCont.push(1);
			} else
				//				pushControl1b(0);
				stackCont.push(0);

		}

		lossFunc += fabs(fapprox - data[tid*(d+1)+d]);

		if (fapprox - data[tid*(d+1) + d] >= 0.0)
			//			pushControl1b(0);
			stackCont.push(0);
		else
			//			pushControl1b(1);
			stackCont.push(1);

		stackReal.push(kernelSum);
		//		pushReal4(kernelSum);
	}
	lossFuncb = *resultb;
	*resultb = 0.0;
	//	*kernelValTableb = 0.0;
	for (int tid = N-1; tid > -1; --tid) {
		double kernelSum;
		double kernelSumb = 0.0;
		//		double fabs0;
		double fabs0b;
		//		double fapprox;
		double fapproxb = 0.0;
		kernelSum = stackReal.top(); stackReal.pop();
		//		popReal4(&kernelSum);
		fabs0b = lossFuncb;
		branch = stackCont.top(); stackCont.pop();
		if (branch == 0)
			fapproxb = fabs0b;
		else
			fapproxb = -fabs0b;
		kernelSumb = 0.0;
		for (int i = N-1; i > -1; --i) {
			branch = stackCont.top(); stackCont.pop();
			if (branch != 0) {
				int indxKernelValTable;
				float tempb;
				indxKernelValTable = stackInt.top(); stackInt.pop();
				//				popInteger4(&indxKernelValTable);
				tempb = data[i*(d+1)+d]*fapproxb/kernelSum;
				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + tempb;
				kernelSumb = kernelSumb - kernelValTable[indxKernelValTable]*tempb/kernelSum;
			}
		}
		for (int i = N-1; i > -1; --i) {
			branch = stackCont.top(); stackCont.pop();
			if (branch != 0) {
				int indxKernelValTable;
				//				popInteger4(&indxKernelValTable);
				indxKernelValTable = stackInt.top(); stackInt.pop();
				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + kernelSumb;
			}
		}
	}


	*result = lossFunc;

#if 0
	printf("kernelValTableb = \n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++) {
			printf("%10.7f ", kernelValTableb[i*N+j]);

		}

		printf("\n");
	}
#endif


}



//void calculateLossKernelL1(codi::RealReverse* result, double *data,codi::RealReverse *kernelValTable, int N, int d){
//
//
//	codi::RealReverse lossFunc = 0.0;
//
//	for(int tid=0; tid<N; tid++){
//
//
//
//		codi::RealReverse kernelSum = 0.0;
//
//		for(int i=0; i<N; i++){
//
//			if(tid != i){
//
//				int indxKernelValTable;
//				if(i<tid) {
//
//
//					indxKernelValTable = i*N+tid;
//
//				}
//				else{
//
//					indxKernelValTable = tid*N+i;
//
//				}
//
//				kernelSum += kernelValTable[indxKernelValTable];
//
//			}
//
//
//
//		}
//
//		codi::RealReverse fapprox=0.0;
//		for(int i=0; i<N; i++){
//
//			if(tid != i){
//				int indxKernelValTable;
//
//				if(i<tid) {
//
//					indxKernelValTable = i*N+tid;
//
//				}
//				else{
//
//					indxKernelValTable = tid*N+i;
//
//				}
//
//				fapprox += (kernelValTable[indxKernelValTable]/kernelSum)* data[i*(d+1)+d];
//
//			}
//
//
//
//
//		}
//
//		lossFunc += fabs(fapprox - data[tid*(d+1)+d]);
//
//	}
//
//	*result = lossFunc;
//
//}








void calculateLossKernelL2(double *result, double *X, double *kernelValTable, int N, int d){


	double lossFunc = 0.0;

	for(int tid=0; tid<N; tid++){


		double kernelSum = 0.0;

		for(int i=0; i<N; i++){

			if(tid != i){

				int indxKernelValTable;
				if(i<tid) {


					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				kernelSum += kernelValTable[indxKernelValTable];

			}



		}

		double fapprox=0.0;
		for(int i=0; i<N; i++){

			if(tid != i){
				int indxKernelValTable;

				if(i<tid) {

					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				fapprox += (kernelValTable[indxKernelValTable]/kernelSum)* X[i*(d+1)+d];
#if 1
				printf("weight = %10.7f f = %10.7f\n", (kernelValTable[indxKernelValTable]/kernelSum),X[i*(d+1)+d]);
#endif

			}


		}


		lossFunc += (fapprox - X[tid*(d+1)+d]) * (fapprox - X[tid*(d+1)+d]);
#if 1
		printf("ftilde = %10.7f fexact = %10.7f\n", fapprox,X[tid*(d+1)+d]);
#endif




	}

	*result = lossFunc;

}







void calculateLossKernelL2_b(double *result, double *resultb, double *X, double *kernelValTable, double *kernelValTableb, int N, int d) {

	stack<int> stackInt;
	stack<int> stackCont;
	stack<double> stackReal;

	for(int i=0; i<N*N; i++) {
		kernelValTableb[i]=0.0;
	}


	double lossFunc = 0.0;
	double lossFuncb = 0.0;
	int branch;
	for (int tid = 0; tid < N; ++tid) {
		double kernelSum = 0.0;
		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				kernelSum = kernelSum + kernelValTable[indxKernelValTable];
				stackInt.push(indxKernelValTable);
				//				pushInteger4(indxKernelValTable);
				//				pushControl1b(1);
				stackCont.push(1);
			} else
				stackCont.push(0);
			//				pushControl1b(0);


		}
		double fapprox = 0.0;
		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				fapprox = fapprox + kernelValTable[indxKernelValTable]/
						kernelSum*X[i*(d+1)+d];
#if 0
				printf("weight = %10.7f f = %10.7f\n", (kernelValTable[indxKernelValTable]/kernelSum),X[i*(d+1)+d]);
#endif
				stackInt.push(indxKernelValTable);
				//				pushInteger4(indxKernelValTable);
				//				pushControl1b(1);
				stackCont.push(1);
			} else
				//				pushControl1b(0);
				stackCont.push(0);

		}

		lossFunc += (fapprox - X[tid*(d+1)+d]) * (fapprox - X[tid*(d+1)+d]);
#if 0
		printf("ftilde = %10.7f fexact = %10.7f\n", fapprox,X[tid*(d+1)+d]);
#endif
		//		pushReal4(kernelSum);
		//		pushReal4(fapprox);

		stackReal.push(kernelSum);
		stackReal.push(fapprox);

	}


	lossFuncb = *resultb;
	*resultb = 0.0;
	*kernelValTableb = 0.0;
	for (int tid = N-1; tid > -1; --tid) {
		double kernelSum;
		double kernelSumb = 0.0;
		double fapprox;
		double fapproxb = 0.0;

		fapprox = stackReal.top(); stackReal.pop();
		kernelSum = stackReal.top(); stackReal.pop();

		//		popReal4(&fapprox);
		//		popReal4(&kernelSum);
		fapproxb = 2*(fapprox-X[tid*(d+1)+d])*lossFuncb;
		kernelSumb = 0.0;
		for (int i = N-1; i > -1; --i) {
			//			popControl1b(&branch);
			branch = stackCont.top(); stackCont.pop();
			if (branch != 0) {
				int indxKernelValTable;
				double tempb;
				//				popInteger4(&indxKernelValTable);
				indxKernelValTable = stackInt.top(); stackInt.pop();
				tempb = X[i*(d+1)+d]*fapproxb/kernelSum;
				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + tempb;
				kernelSumb = kernelSumb - kernelValTable[indxKernelValTable]*tempb/kernelSum;
			}
		}
		for (int i = N-1; i > -1; --i) {
			//			popControl1b(&branch);
			branch = stackCont.top(); stackCont.pop();
			if (branch != 0) {
				int indxKernelValTable;
				indxKernelValTable = stackInt.top(); stackInt.pop();
				//				popInteger4(&indxKernelValTable);
				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + kernelSumb;
			}
		}
	}


	*result = lossFunc;
#if 0
	printf("kernelValTableb = \n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++) {
			printf("%10.7f ", kernelValTableb[i*N+j]);

		}

		printf("\n");
	}
#endif



}


//void calculateLossKernelL2(codi::RealReverse* result, double *X,codi::RealReverse *kernelValTable, int N, int d){
//
//
//	codi::RealReverse lossFunc = 0.0;
//
//	for(int tid=0; tid<N; tid++){
//
//
//		codi::RealReverse kernelSum = 0.0;
//
//		for(int i=0; i<N; i++){
//
//			if(tid != i){
//
//				int indxKernelValTable;
//				if(i<tid) {
//
//
//					indxKernelValTable = i*N+tid;
//
//				}
//				else{
//
//					indxKernelValTable = tid*N+i;
//
//				}
//
//				kernelSum += kernelValTable[indxKernelValTable];
//
//			}
//
//
//
//		}
//
//		codi::RealReverse fapprox=0.0;
//		for(int i=0; i<N; i++){
//
//			if(tid != i){
//				int indxKernelValTable;
//
//				if(i<tid) {
//
//					indxKernelValTable = i*N+tid;
//
//				}
//				else{
//
//					indxKernelValTable = tid*N+i;
//
//				}
//
//				fapprox += (kernelValTable[indxKernelValTable]/kernelSum)* X[i*(d+1)+d];
//#if 0
//				printf("weight = %10.7f f = %10.7f\n", (kernelValTable[indxKernelValTable]/kernelSum),X[i*(d+1)+d]);
//#endif
//
//			}
//
//
//		}
//
//
//		lossFunc += (fapprox - X[tid*(d+1)+d]) * (fapprox - X[tid*(d+1)+d]);
//#if 0
//		printf("ftilde = %10.7f fexact = %10.7f\n", fapprox,X[tid*(d+1)+d]);
//#endif
//
//
//
//
//	}
//
//	*result = lossFunc;
//
//}



/*
 * calculates the loss function for the given training data
 *
 * @param[in] data : first vector
 * @param[in] input : Matrix M + sigma
 * @param[in] N : size of the training data
 * @param[in] d : problem dimension
 * @param[in] lossFunType :
 * @param[out] result
 *
 * */


void calcLossFunCPU(double *result, double *input, double *data,int N, int d,int lossFunType){


	double LT[d][d];
	double L[d][d];
	double M[d*d];





	for (int i = 0; i < d; i++)
		for (int j = 0; j < d; j++) {
			L[i][j]=input[i*d+j];

		}


	double sigma = input[d*d];



	for (int i = 0; i < d; i++)
		for (int j = 0; j < d; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < d; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}


	for(int i = 0; i < d; ++i)
		for(int j = 0; j < d; ++j)
		{
			M[i*d+j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < d; ++i)
		for(int j = 0; j < d; ++j)
			for(int k = 0; k < d; ++k)
			{
				M[i*d+j] += L[i][k] * LT[k][j];

			}



	double *kernelValTable = new double[N*N];

	calculateKernelValues(data, kernelValTable, M, sigma, N, d);



	switch ( lossFunType )
	{
	case L1_LOSS_FUNCTION:
		calculateLossKernelL1(result,data,kernelValTable, N, d);
		break;
	case L2_LOSS_FUNCTION:
		calculateLossKernelL2(result, data,kernelValTable, N,d);
		break;
	default:
		printf("Error: Unknown lossFunType at %s, line %d\n",__FILE__, __LINE__);
		exit(-1);
	}





#if 0
	printf("loss = %10.7f\n", *result);
#endif

	delete [] kernelValTable;

}

/*
 * calculates the loss function for the given training data (CodiPack validation version)
 *
 * @param[in] data : first vector
 * @param[in] input : Matrix M + sigma
 * @param[out] inputb : sensitivites w.r.t. Matrix M + sigma
 * @param[in] N : size of the training data
 * @param[in] d : problem dimension
 * @param[in] lossFunType :
 * @param[out] result
 *
 * */

//void calcLossFunCPUCodi(double *result, double *input, double *inputb, double *data,int N, int d,int lossFunType){
//
//
//	codi::RealReverse *inputcodi = new codi::RealReverse[d*d+1];
//
//	codi::RealReverse LT[d][d];
//	codi::RealReverse L[d][d];
//	codi::RealReverse M[d*d];
//
//	codi::RealReverse *kernelValTable = new codi::RealReverse[N*N];
//
//
//	/* activate tape and register input */
//
//	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
//	tape.setActive();
//
//
//	for (int i = 0; i < d*d+1; i++) {
//		inputcodi[i] = input[i];
//		tape.registerInput(inputcodi[i]);
//
//	}
//
//
//
//	codi::RealReverse resultCodi;
//
//
//	for (int i = 0; i < d; i++)
//		for (int j = 0; j < d; j++) {
//			L[i][j]=inputcodi[i*d+j];
//
//		}
//
//
//	codi::RealReverse sigma = inputcodi[d*d];
//
//
//
//	for (int i = 0; i < d; i++)
//		for (int j = 0; j < d; j++) {
//
//			LT[i][j]=0.0;
//		}
//
//
//
//
//	for (int i = 0; i < d; i++) {
//		for (int j = 0; j <= i; j++){
//
//			LT[j][i] = L[i][j];
//		}
//
//
//	}
//
//
//	for(int i = 0; i < d; ++i)
//		for(int j = 0; j < d; ++j)
//		{
//			M[i*d+j]=0;
//		}
//
//	/* Multiplying matrix L and LT and storing in M */
//	for(int i = 0; i < d; ++i)
//		for(int j = 0; j < d; ++j)
//			for(int k = 0; k < d; ++k)
//			{
//				M[i*d+j] += L[i][k] * LT[k][j];
//
//			}
//
//	calculateKernelValues(data, kernelValTable, M, sigma, N, d);
//
//	if(lossFunType == L1_LOSS_FUNCTION){
//
//		calculateLossKernelL1(&resultCodi,data,kernelValTable, N, d);
//
//	}
//
//	if(lossFunType == L2_LOSS_FUNCTION){
//
//		calculateLossKernelL2(&resultCodi, data,kernelValTable, N,d);
//
//	}
//
//
//
//	tape.registerOutput(resultCodi);
//
//	tape.setPassive();
//	resultCodi.setGradient(1.0);
//	tape.evaluate();
//
//	for (int i = 0; i < d*d+1; i++) {
//
//
//		inputb[i] = inputcodi[i].getGradient();
//
//	}
//
//
//
//
//	*result = resultCodi.getValue();
//
//#if 1
//	printf("loss = %10.7f\n", *result);
//#endif
//
//#if 1
//
//
//	printf("Lb =\n");
//	for (int i = 0; i < d; i++){
//		for (int j = 0; j < d; j++) {
//			printf("%10.7f ",inputb[i*d+j]);
//
//		}
//
//		printf("\n");
//	}
//
//	printf("sigmab = %10.7f\n",inputb[d*d]);
//
//#endif
//
//	tape.reset();
//
//
//	delete [] kernelValTable;
//
//}


void calcLossFunCPU_b(double *result, double *input, double *inputb, double *data,int N, int d,int lossFunType){


	double LT[d][d];
	double L[d][d];
	double LTb[d][d];
	double Lb[d][d];
	double M[d*d];
	double Mb[d*d];

	for (int i = 0; i < d*d+1; i++) {

		inputb[i] = 0.0;
	}



	for (int i = 0; i < d; i++)
		for (int j = 0; j < d; j++) {
			Lb[i][j]=0.0;
			L[i][j]=input[i*d+j];

		}


	double sigma = input[d*d];
	double sigmab = 0.0;


#if 0
	printf("Training data = \n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < d+1; j++) {

			printf("%10.7f ", data[i*(d+1)+j]);

		}
		printf("\n");
	}

#endif



#if 0
	printf("L = \n");

	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++) {

			printf("%10.7f ", L[i][j]);

		}
		printf("\n");
	}

#endif




	for (int i = 0; i < d; i++)
		for (int j = 0; j < d; j++) {

			LT[i][j]=0.0;
			LTb[i][j]=0.0;
		}




	for (int i = 0; i < d; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}
#if 0
	printf("LT = \n");

	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++) {

			printf("%10.7f ", LT[i][j]);

		}
		printf("\n");
	}

#endif

	for(int i = 0; i < d; ++i)
		for(int j = 0; j < d; ++j)
		{
			M[i*d+j]=0;
			Mb[i*d+j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < d; ++i)
		for(int j = 0; j < d; ++j)
			for(int k = 0; k < d; ++k)
			{
				M[i*d+j] += L[i][k] * LT[k][j];

			}
#if 0
	printf("M = \n");

	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++) {

			printf("%10.7f ", M[i*d+j]);

		}
		printf("\n");
	}

#endif


	double *kernelValTable = new double[N*N];
	double *kernelValTableb = new double[N*N];

	double resultb = 1.0;

	calculateKernelValues(data, kernelValTable, M, sigma, N, d);



	switch ( lossFunType )
	{
	case L1_LOSS_FUNCTION:
		calculateLossKernelL1_b(result, &resultb, data, kernelValTable, kernelValTableb, N, d);
		break;
	case L2_LOSS_FUNCTION:
		calculateLossKernelL2_b(result, &resultb, data, kernelValTable, kernelValTableb, N, d);
		break;
	default:
		printf("Error: Unknown lossFunType at %s, line %d\n",__FILE__, __LINE__);
		exit(-1);
	}


#if 0
	printf("loss = %10.7f\n", *result);
#endif

	calculateKernelValues_b(data, kernelValTable, kernelValTableb, M, Mb, sigma, &sigmab, N, d);



	for (int i = d-1; i > -1; --i)
		for (int j = d-1; j > -1; --j)
			for (int k = d-1; k > -1; --k) {
				Lb[i][k] = Lb[i][k] + LT[k][j]*Mb[i*d+j];
				LTb[k][j] = LTb[k][j] + L[i][k]*Mb[i*d+j];
			}

	for (int i = d-1; i > -1; --i) {
		for (int j = i; j > -1; --j) {
			Lb[i][j] = Lb[i][j] + LTb[j][i];
			LTb[j][i] = 0.0;
		}
	}


	for (int i = d-1; i > -1; --i)
		for (int j = d-1; j > -1; --j) {
			inputb[i*d + j] = inputb[i*d + j] + Lb[i][j];
			Lb[i][j] = 0.0;
		}


	inputb[d*d] = sigmab;

#if 0
	printf("Lb =\n");
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++) {
			printf("%10.7f ",inputb[i*d+j]);

		}

		printf("\n");
	}

	printf("sigmab = %10.7f\n",sigmab);

#endif

#if 0

	/* validation part */

	double eps = 0.00001;
	double f0 = *result;
	double fp;
	double fdres[d*d+1];

	for(int i=0; i<d*d+1; i++){

		input[i]+=eps;

		calcLossFunCPU(&fp, input, data, N, d, lossFunType);
		fdres[i] = (fp - f0)/eps;
		input[i]-=eps;

	}


	printf("fd results =\n");
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++) {
			printf("%10.7f ",fdres[i*d+j]);

		}

		printf("\n");
	}



	printf("sigmab (fd) = %10.7f\n",fdres[d*d]);
	printf("Lb =\n");
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++) {
			printf("%10.7f ",inputb[i*d+j]);

		}

		printf("\n");
	}

	printf("sigmab = %10.7f\n",sigmab);

	printf("CodiPack results...\n");
	calcLossFunCPUCodi(result, input, inputb, data, N, d,lossFunType);



	exit(1);

#endif

	delete [] kernelValTable;
	delete [] kernelValTableb;


}





double kernelRegressorNotNormalized(mat &X, vec &y, rowvec &xp, vec& xmax, vec &xmin, mat &M, double sigma) {

	int N = y.size();

	vec kernelVal(N);
	vec weight(N);

	rowvec xpNormalized;

	/* first normalize xp */

	for (unsigned int j = 0; j < xp.size(); j++) {

		xpNormalized(j) = (1.0/xp.size())*(xp(j) - xmin(j)) / (xmax(j) - xmin(j));
	}

	/* calculate the kernel values */

	double kernelSum = 0.0;
	for (int i = 0; i < N; i++) {

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi, xpNormalized, sigma, M);
		kernelSum += kernelVal(i);
	}

	double yhat = 0.0;
	for (int i = 0; i < N; i++) {

		weight(i) = kernelVal(i) / kernelSum;
		yhat += y(i) * weight(i);
#if 0
		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );
#endif
	}

	return yhat;

}


/*
 * return kernel regression estimate
 * @param[in] X: sample input values (normalized)
 * @param[in] xp: point to be estimated
 * @param[in] M : Mahalanobis matrix
 * @param[in] sigma:  bandwidth parameter
 * @param[in] y: functional values at sample locations(normalized)
 *
 * */

double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma) {
#if 0
	printf("xp: "); xp.print();
#endif
	int N = y.size();

	fvec kernelVal(N);
	fvec weight(N);


	double kernelSum = 0.0;
	for (int i = 0; i < N; i++) {

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi, xp, sigma, M);
		kernelSum += kernelVal(i);
	}

	if(kernelSum < 10E-06){

		printf("Warning: kernelSum is too small! KernelSum = %15.10f\n",kernelSum);


	}



	double yhat = 0.0;
	for (int i = 0; i < N; i++) {

		weight(i) = kernelVal(i) / kernelSum;
		yhat += y(i) * weight(i);
#if 0
		if(weight(i) > 0.05){
			X.row(i).print();
			printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );

		}
#endif
	}



	return yhat;

}





/*
 * return kernel regression estimate with gradient data
 * @param[in] X: sample input values (normalized)
 * @param[in] grad: sample gradient values (normalized)
 * @param[in] xp: point to be estimated
 * @param[in] M : Mahalanobis matrix
 * @param[in] sigma:  bandwidth parameter
 * @param[in] y: functional values at sample locations(normalized)
 *
 * */


double kernelRegressor(mat &X, vec &y, mat &grad, rowvec &xp, mat &M, double sigma) {

	int N = y.size();

	vec kernelVal(N);
	vec weight(N);
	double kernelSum = 0.0;
	double yhat = 0.0;




	/* first evaluate the kernel sum */
	for (int i = 0; i < N; i++) {

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi, xp, sigma, M);
		kernelSum += kernelVal(i);


	}

	rowvec xdiff(xp.size());

	for (int i = 0; i < N; i++) {

		rowvec xi = X.row(i);

		for(unsigned int j=0; j<xp.size(); j++) xdiff(j) = xp(j) -xi(j);

		xdiff.print();

#if 0
		printf("xp =\n");
		xp.print();
		printf("xi =\n");
		xi.print();
		printf("xdiff =\n");
		xdiff.print();
		printf("grad =\n");
		grad.row(i).print();
#endif


		double gradTerm = dot(xdiff,grad.row(i));
#if 0
		printf("gradTerm = %10.7f\n",gradTerm);
		printf("y = %10.7f\n",y(i));
#endif



		//		weight(i) = kernelVal(i) / kernelSum;
		yhat += (y(i) + gradTerm) * weight(i);
		//		yhat += (y(i) ) * weight(i);
#if 0
		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );
#endif
	}


	printf("yhat = %10.7f\n",yhat);
	return yhat;

}


///*
// * return kernel regression estimate with gradient data
// * @param[in] X: sample input values (normalized)
// * @param[in] XnotNormalized: sample input values
// * @param[in] grad: sample gradient values (not normalized)
// * @param[in] xp: point to be estimated
// * @param[in] M : Mahalanobis matrix
// * @param[in] sigma:  bandwidth parameter
// * @param[in] y: functional values at sample locations
// *
// * */
//
//
//float kernelRegressorNotNormalized(fmat &X,
//		fmat &XnotNormalized,
//		fvec &y,
//		fmat &grad,
//		frowvec &xp,
//		fvec &xmin,
//		fvec &xmax,
//		fmat &M,
//		float sigma) {
//
//
//	/* number of samples */
//	int N = y.size();
//	int d = xp.size();
//
//	fvec kernelVal(N);
//	fvec weight(N);
//
//	frowvec xpNormalized(d);
//
//	/* first normalize xp */
//
//	for (int j = 0; j < d; j++) {
//
//		xpNormalized(j) = (1.0/d)*(xp(j) - xmin(j)) / (xmax(j) - xmin(j));
//	}
//
//
//	float kernelSum = 0.0;
//
//
//	frowvec xi(d);
//	frowvec xdiff(d);
//
//	/* first evaluate the kernel sum */
//	for (int i = 0; i < N; i++) {
//
//		xi = X.row(i);
//
//		kernelVal(i) = gaussianKernel(xi, xpNormalized, sigma, M);
//		kernelSum += kernelVal(i);
//	}
//
//
//
//	float yhat = 0.0;
//
//	for (int i = 0; i < N; i++) {
//
//
//		xi = XnotNormalized.row(i);
//		for(int j=0; j<d; j++) {
//
//			xdiff(j) = xp(j) -xi(j);
//		}
//
//
//		float gradTerm = dot(xdiff,grad.row(i));
//
//		weight(i) = kernelVal(i) / kernelSum;
//		yhat += (y(i) + gradTerm) * weight(i);
//#if 0
//		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );
//#endif
//	}
//
//	return yhat;
//
//}


/*
 * return kernel regression estimate with gradient data
 * @param[in] X: sample input values (normalized)
 * @param[in] XnotNormalized: sample input values
 * @param[in] grad: sample gradient values (not normalized)
 * @param[in] xp: point to be estimated
 * @param[in] M : Mahalanobis matrix
 * @param[in] sigma:  bandwidth parameter
 * @param[in] y: functional values at sample locations
 *
 * */


double kernelRegressorNotNormalized(mat &X,
		mat &XnotNormalized,
		vec &y,
		mat &grad,
		rowvec &xp,
		vec &xmin,
		vec &xmax,
		mat &M,
		double sigma) {


	/* number of samples */
	int N = y.size();
	int d = xp.size();

	vec kernelVal(N);
	vec weight(N);

	rowvec xpNormalized(d);

	/* first normalize xp */

	for (int j = 0; j < d; j++) {

		xpNormalized(j) = (1.0/d)*(xp(j) - xmin(j)) / (xmax(j) - xmin(j));
	}
#if 1
	printf("xpNormalized:\n");
	xpNormalized.print();
#endif
	double kernelSum = 0.0;


	rowvec xi(d);
	rowvec xdiff(d);

	/* first evaluate the kernel sum */
	for (int i = 0; i < N; i++) {

		xi = X.row(i);
#if 1
		printf("xi:\n");
		xi.print();
#endif

		kernelVal(i) = gaussianKernel(xi, xpNormalized, sigma, M);
#if 1
		printf("kernelVal(%d) = %10.7f\n",i,kernelVal(i));
#endif
		kernelSum += kernelVal(i);
	}



	double yhat = 0.0;

	for (int i = 0; i < N; i++) {


		xi = XnotNormalized.row(i);
		for(int j=0; j<d; j++) {

			xdiff(j) = xp(j) -xi(j);
		}


		double gradTerm = dot(xdiff,grad.row(i));

		weight(i) = kernelVal(i) / kernelSum;
		yhat += (y(i) + gradTerm) * weight(i);
#if 1
		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );
#endif
	}

	return yhat;

}

//int trainMahalanobisDistanceBruteForce(mat &L, mat &data, double &sigma, double yTrainingMax, int lossFunType, int batchsize, int ntrials) {
//
//#if 1
//
//	printf("trainMahalanobisDistanceBruteForce...\n");
//#endif
//
//	unsigned int d = L.n_cols;
//
//	unsigned int Ldim = d*d;
//
//#if 1
//	printf("Ldim = %d\n",Ldim);
//
//	printf("Data (normalized) with ymax = %10.7f =\n",yTrainingMax);
//	data.print();
//
//#endif
//
//	/* lower diagonal matrix Lbest to keep the best L*/
//	mat bestL(d,d);
//	bestL.fill(0.0);
//
//	double bestsigma = 0.0;
//
//
//	/* divide the data set into training and validation sets */
//
//	unsigned int N = data.n_rows;
//
//
//	/* size of the validation set, default to one fifth */
//	unsigned int NvalidationSet = N/5;
//	unsigned int Ntraining = N - NvalidationSet;
//#if 1
//	printf("N = %d, Ntraining = %d, Nvalidation =%d\n", N,Ntraining, NvalidationSet);
//#endif
//	mat dataTraining      = data.submat( 0, 0, Ntraining-1, d );
//	mat dataValidation    = data.submat( Ntraining, 0, N-1, d );
//
//
//	mat XValidation = dataValidation.submat(0,0,NvalidationSet-1,d-1);
//	vec yValidation = dataValidation.col(d);
//
//	mat XTraining = dataTraining.submat(0,0,Ntraining-1,d-1);
//	vec yTraining = dataTraining.col(d);
//
//
//
//
//	double optGenError = LARGE;
//	printf("ntrials = %d\n", ntrials);
//	/* optimization loop */
//
//
//
//	for(int ii=0; ii<100000; ii++){
//
//
//
//
//#if 0
//		if(ii > ntrials)
//
//			printf("iterTrial = %d, ntrials = %d\n", ii,ntrials);
//#endif
//
//		for (int i = 0; i < d; i++)
//			for (int j = 0; j <= i; j++) {
//
//				L(i,j) = RandomFloat(0.0,1.0);
//
//			}
//
//		sigma = RandomFloat(0.0,1.0);
//
//
//		if(ii == 0){
//
//			sigma = 0.1;
//			L = eye<fmat>(d,d);
//
//		}
//
//
//
//		fmat M = L*trans(L);
//
//
//
//
//#if 0
//		printf("L = \n");
//		L.print();
//		printf("M = \n");
//		M.print();
//		printf("sigma = %10.7f\n", sigma);
//#endif
//
//
//		float genError = 0.0;
//
//		for(int i=0;i <NvalidationSet; i++){
//
//			frowvec xp = XValidation.row(i);
//			float ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax;
//			float yexact = yValidation(i)*yTrainingMax;
//
//#if 0
//			printf("i= %d, x:\n",i);
//			xp.print();
//			printf("ytilde = %10.7f, yexact = %10.7f\n",ytilde,yexact);
//#endif
//
//
//
//			if( lossFunType == L1_LOSS_FUNCTION) genError += fabs(yexact-ytilde);
//			if( lossFunType == L2_LOSS_FUNCTION) genError += (yexact-ytilde)*(yexact-ytilde);
//
//		}
//
//		genError = genError/NvalidationSet;
//
//		if(ii == 0){
//			printf("Generalization error (for M=I) = %10.7f\n",genError);
//
//		}
//
//
//#if 0
//		printf("Generalization error = %10.7f\n",genError);
//#endif
//		if(genError < optGenError) {
//
//#if 1
//			printf("Better L has been found, updating L...\n");
//			printf("Generalization error = %10.7f\n",genError);
//
//			printf("L = \n");
//			L.print();
//			printf("M = \n");
//			M.print();
//			printf("sigma = %10.7f\n", sigma);
//
//#endif
//			bestL = L;
//			bestsigma = sigma;
//			optGenError = genError;
//
//
//		}
//
//
//
//
//
//
//
//
//	}
//
//
//
//	L = bestL;
//	sigma = bestsigma;
//	printf("L = \n");
//	L.print();
//	printf("optimization finished...\n");
//	return 0;
//
//}

//int trainMahalanobisDistance(mat &L, mat &data, double &sigma, double &wSvd, double &w12,
//		unsigned int max_cv_iter, unsigned int lossFunType, unsigned int batchsize, unsigned int nepochs) {
//
//
//	double learning_rateM = 0.00001;
//	double learning_rateSigma = learning_rateM * 0.01;
//
//	bool trainWithSvdFlag = false;
//
//	if(wSvd > 0.0) {
//
//		trainWithSvdFlag = true;
//	}
//
//
//	//	const double alpha = 0.9;
//
//	unsigned int d = L.n_cols;
//
//	/* Ldim is the number of entries in the L matrix */
//	unsigned int Ldim = d*d;
//
//	/* lower diagonal matrix Lbest to keep the best L*/
//	mat bestL(d,d);
//	bestL.fill(0.0);
//
//	double bestsigma = 0.0;
//
//
//	/* divide the data set into training and validation sets */
//
//	unsigned int N = data.n_rows;
//
//
//	/* size of the validation set, default to one fifth */
//	unsigned int NvalidationSet = N/5;
//	unsigned int Ntraining = N - NvalidationSet;
//#if 1
//	printf("Training the Mahalanobis distance...\n");
//	printf("Mini batch size = %d\n",batchsize);
//	printf("Number of epochs = %d\n",nepochs);
//	printf("number of cross-validation iterations = %d\n",max_cv_iter);
//	printf("number of training samples (core) = %d\n",Ntraining);
//	printf("number of validation samples      = %d\n",NvalidationSet);
//
//#endif
//
//
//
//	/* do not allow that batch size will be greater than number of training samples */
//	if (batchsize > Ntraining){
//
//		batchsize = Ntraining;
//	}
//
//	/* divide the data into two sets */
//
//	mat dataTraining      = data.submat( 0, 0, Ntraining-1, d );
//	mat dataValidation    = data.submat( Ntraining, 0, N-1, d );
//
//
//	mat XValidation = dataValidation.submat(0,0,NvalidationSet-1,d-1);
//	vec yValidation = dataValidation.col(d);
//
//	mat XTraining = dataTraining.submat(0,0,Ntraining-1,d-1);
//	vec yTraining = dataTraining.col(d);
//
//	mat dataMiniBatch(batchsize, d+1);
//
//
//#if 0
//
//	printf("Training data set = \n");
//	dataTraining.print();
//
//	printf("Validation data set = \n");
//	dataValidation.print();
//#endif
//
//
//#if 0
//	printf("XTraining = \n");
//	XTraining.print();
//	printf("yTraining = \n");
//	yTraining.print();
//#endif
//
//#if 0
//	printf("XValidation = \n");
//	XValidation.print();
//	printf("yValidation = \n");
//	yValidation.print();
//#endif
//
//	vec wSvdtrial(max_cv_iter);
//	vec w12trial(max_cv_iter);
//
//
//	if(max_cv_iter !=1){
//
//		for(unsigned int i=0; i<max_cv_iter; i++){
//
//			wSvdtrial(i) = pow(10.0,generateRandomDouble(-2,0.0));
//			w12trial(i) = pow(10.0,generateRandomDouble(-5,0.0));
//		}
//#if 0
//		printf("wSvdtrial = \n");
//		trans(wSvdtrial).print();
//		printf("w12trial = \n");
//		trans(w12trial).print();
//#endif
//
//	}
//
//	/* auxilliary vectors */
//
//	double *inputVec = new double[Ldim+1]();
//	//	double *inputVecVel = new double[Ldim+1]();
//	double *inputVecLocalBest = new double[Ldim+1]();
//	double *inputVecb = new double[Ldim+1]();
//	double *inputVecRegb = new double[Ldim]();
//	double *gradientVec = new double[Ldim+1]();
//
//
//	double *dataVecTraining = new double[batchsize*(d+1)]();
//
//
//
//
//	double optGenError = 10E14;
//
//	/* cross validation loop to tune the weights for the regularization parameters */
//	for(unsigned int iter_cv=0; iter_cv< max_cv_iter; iter_cv++){
//
//
//		if(max_cv_iter !=1){
//
//			if(trainWithSvdFlag){
//				wSvd = wSvdtrial(iter_cv);
//			}
//			else{
//
//				wSvd = 0.0;
//			}
//			w12 =  w12trial(iter_cv);
//		}
//#if 1
//		printf("Outer iteration = %d\n",iter_cv);
//		printf("wSvd = %10.7f, w12 = %10.7f\n",wSvd,w12);
//#endif
//
//		/* initialize the L matrix and sigma => everything is saved in the vector "inputVec" */
//
//		for (unsigned int i = 0; i < d; i++)
//			for (unsigned int j = 0; j < d; j++) {
//
//				inputVec[i*d+j] = 0.0;
//			}
//
//		for (unsigned int i = 0; i < d; i++) {
//
//			for (unsigned int j = 0; j <= i; j++) {
//
//				if(i ==j) { /* main diagonal */
//
//					inputVec[i*d+j] = 1.0+ generateRandomDouble(-0.1,0.1);
//				}
//				else {
//
//					inputVec[i*d+j] = generateRandomDouble(0.0,0.1);
//				}
//			}
//		}
//
//		/* assign sigma to a small value between 0 and 0.1 */
//		inputVec[Ldim] = 0.05+ generateRandomDouble(-0.001,0.001);
//#if 1
//		printf("Initial values of L:\n");
//
//		for (unsigned int i = 0; i < d; i++){
//			for (unsigned int j = 0; j < d; j++) {
//
//				printf("%10.7f ",inputVec[i*d+j]);
//			}
//			printf("\n");
//		}
//
//		printf("Initial sigma = %10.7f\n",inputVec[Ldim]);
//
//#endif
//		double lossVal, regTerm;
//		double objFunVal;
//		lossVal = 0.0;
//
//		double objectiveFunLocalBest = 10E14;
//
//
//
//		/* optimization loop */
//
//		for(unsigned int opt_iter=0 ; opt_iter < nepochs; opt_iter++){
//
//
//			dataTraining = shuffle(dataTraining);
//
//			/*get the mini batch */
//			dataMiniBatch = dataTraining.submat(0,0,batchsize-1,d);
//
//#if 0
//			printf("dataMiniBatch = \n");
//			dataMiniBatch.print();
//#endif
//
//
//
//
//#if 0
//			printf("copying training data...\n");
//#endif
//			for (unsigned int i = 0; i < batchsize; i++) {
//
//				for (unsigned int j = 0; j < d+1; j++) {
//
//					dataVecTraining[i*(d+1)+j ] = dataMiniBatch (i, j);
//				}
//			}
//#if 0
//			printf("data copied = \n");
//
//			for (int i = 0; i < batchsize; i++) {
//
//				for (int j = 0; j < d+1; j++) {
//
//					printf("%10.7f ",dataVecTraining[i*(d+1)+j ]);
//				}
//				printf("\n");
//			}
//
//#endif
//
//
//
//
//
//			/* init sensitivities to zero */
//			for(unsigned int i=0;i<Ldim+1;i++) {
//
//				inputVecb[i] = 0.0;
//			}
//
//			calcLossFunCPU_b(&lossVal,inputVec,inputVecb, dataVecTraining,batchsize,d, lossFunType);
//#if 0
//			printf("Loss (CPU Version)= %10.7f\n", lossVal);
//#endif
//			for(unsigned int i=0;i<Ldim+1;i++) {
//
//				gradientVec[i]=inputVecb[i];
//			}
//#if 0
//			printf("calculating the regularization term...\n");
//#endif
//			for(unsigned int i=0;i<Ldim;i++) {
//
//				inputVecRegb[i] = 0.0;
//			}
//
//			/* call the adjoint mode of the function to compute the regularization term */
//
//			if(trainWithSvdFlag){
//
//				calcRegTerms(inputVec, inputVecRegb, &regTerm, wSvd, w12, d);
//			}
//			else{
//
//				calcRegTermL12(inputVec, inputVecRegb,&regTerm, w12, d);
//
//			}
//
//
//#if 0
//			printf("gradient of the regularization term = \n");
//
//			for (int i = 0; i < numVar; i++) {
//				for (int j = 0; j < numVar; j++) {
//
//					printf("%10.7f ", inputVecRegb[i*numVar+j]);
//
//				}
//				printf("\n");
//			}
//#endif
//
//
//			/* add the regularization sensitivities to the gradient vector */
//
//			for(unsigned int i=0;i<Ldim;i++) {
//
//				gradientVec[i]+=inputVecRegb[i];
//			}
//
//
//			objFunVal = lossVal + regTerm;
//
//			/* check if gradient has some NaNs*/
//			for(unsigned int i=0;i<Ldim;i++) {
//
//				if( gradientVec[i] != gradientVec[i]){
//
//					printf("Error: gradientVec[%d] is NaN!\n",i);
//					exit(1);
//
//				}
//			}
//
//
//
//			/* update L */
//
//			for (unsigned int i = 0; i < d; i++){
//				for (unsigned int j = 0; j <= i; j++) {
//
//					//					inputVec[i*d+j]= inputVec[i*d+j] + inputVecVel[i*d+j];
//					inputVec[i*d+j]= inputVec[i*d+j] -learning_rateM * gradientVec[i*d+j];
//
//				}
//
//			}
//
//			/* do not allow that entries of L will be negative! */
//
//			for (unsigned int i = 0; i < d; i++){
//				for (unsigned int j = 0; j <= i; j++) {
//
//					if ( inputVec[i*d+j] < 0) {
//
//						inputVec[i*d+j] = 10E-6;
//
//					}
//
//
//				}
//
//			}
//
//			/* update sigma */
//			//			inputVec[Ldim]= inputVec[Ldim] + inputVecVel[Ldim];
//			inputVec[Ldim]= inputVec[Ldim] - learning_rateSigma *gradientVec[Ldim];
//
//			if(inputVec[Ldim] <= 0) {
//
//				inputVec[Ldim] = 10E-06;
//			}
//
//
//
//
//			if(objFunVal < objectiveFunLocalBest){
//
//				objectiveFunLocalBest = objFunVal;
//
//				for(unsigned int i=0;i<Ldim+1;i++) {
//
//					inputVecLocalBest[i]=inputVec[i];
//
//				}
//
//			}
//
//
//			if(opt_iter % 100 == 0){
//
//#if 1
//				printf("iter = %d, objective function = %10.7f, Leave One Out Error = %10.7f, Regularization term = %10.7f\n",opt_iter,objFunVal,lossVal, regTerm);
//#endif
//#if 0
//				printf("L = \n");
//
//				for (int i = 0; i < d; i++) {
//					for (int j = 0; j < d; j++) {
//
//						printf("%10.7f ", inputVec[i*d+j]);
//
//					}
//					printf("\n");
//				}
//
//				printf("sigma = %10.7f\n",inputVec[Ldim]);
//#endif
//
//
//
//			}
//
//
//
//
//
//		} /* end of local optimization loop */
//
//
//
//
//		for (unsigned int i = 0; i < d; i++)
//			for (unsigned int j = 0; j < d; j++) {
//
//				L(i,j)= inputVecLocalBest[i*d+j];
//			}
//
//#if 1
//		printf("local optimization result:\n");
//		printf("L = \n");
//		L.print();
//		printf("sigma = %10.7f\n", inputVecLocalBest[Ldim]);
//
//#endif
//		sigma = inputVecLocalBest[Ldim];
//
//
//		mat M = L*trans(L);
//#if 1
//		printf("M = \n");
//		M.print();
//#endif
//
//
//		dataTraining      = data.submat( 0, 0, Ntraining-1, d );
//
//
//		double genError = 0.0;
//
//		for(unsigned int i=0;i <NvalidationSet; i++){
//
//			rowvec xp = XValidation.row(i);
//			double ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma);
//			double yexact = yValidation(i);
//
//#if 0
//			printf("x:\n");
//			xp.print();
//			printf("ytilde = %10.7f, yexact = %10.7f\n",ytilde,yexact);
//#endif
//
//
//
//			if( lossFunType == L1_LOSS_FUNCTION) {
//
//				genError += fabs(yexact-ytilde);
//			}
//			if( lossFunType == L2_LOSS_FUNCTION) {
//
//				genError += (yexact-ytilde)*(yexact-ytilde);
//			}
//
//		}
//
//		genError = genError/NvalidationSet;
//
//#if 0
//		printf("Generalization error = %10.7f\n",genError);
//#endif
//		if(genError < optGenError) {
//
//#if 0
//			printf("Better L has been found, updating L...\n");
//#endif
//			bestL = L;
//			bestsigma = sigma;
//			optGenError = genError;
//
//
//		}
//
//
//
//
//
//
//	} /* end of cv loop */
//
//	L = bestL;
//	sigma = bestsigma;
//
//
//
//	delete[] inputVec;
//	delete[] inputVecLocalBest;
//	delete[] inputVecb;
//	delete[] inputVecRegb;
//	delete[] dataVecTraining;
//	delete[] gradientVec;
//
//	return 0;
//
//
//}



