/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<cassert>


#include "tgek.hpp"
#include "auxiliary_functions.hpp"
#include "kriging_training.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

void TGEKModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.setDirectionalDerivativesOn();
	data.readData(filenameDataInput);

	ifDataIsRead = true;

}



void TGEKModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;

}
void TGEKModel::setNameOfHyperParametersFile(string filename){


}
void TGEKModel::setNumberOfTrainingIterations(unsigned int n){

	numberOfTrainingIterations = n;

}

void TGEKModel::setNumberOfDifferentiatedBasisFunctionsUsed(unsigned int n){

	numberOfDifferentiatedBasisFunctions = n;
}


void TGEKModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(ifNormalized);


	output.printMessage("Initializing the generalized derivative enhanced model...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();


	correlationFunction.setInputSampleMatrix(data.getInputMatrix());

	if(!ifCorrelationFunctionIsInitialized){

		correlationFunction.initialize();
		ifCorrelationFunctionIsInitialized = true;
	}

	numberOfHyperParameters = dim;
	w = ones<vec>(data.getNumberOfSamples() + numberOfDifferentiatedBasisFunctions);
	initializeHyperParameters();

	prepareTrainingDataForTheKrigingModel();

	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);
	auxiliaryModel.setBoxConstraints(data.getBoxConstraints());
	auxiliaryModel.readData();
	auxiliaryModel.normalizeData();
	auxiliaryModel.initializeSurrogateModel();
	auxiliaryModel.setNumberOfTrainingIterations(numberOfTrainingIterations);
	auxiliaryModel.setNumberOfThreads(numberOfThreads);
	auxiliaryModel.initializeSurrogateModel();

	ifInitialized = true;


}
void TGEKModel::printSurrogateModel(void) const{


}
void TGEKModel::printHyperParameters(void) const{

	vec hyperParameters = correlationFunction.getHyperParameters();
	hyperParameters.print("theta = ");


}
void TGEKModel::saveHyperParameters(void) const{



}
void TGEKModel::loadHyperParameters(void){


}

vec  TGEKModel::getHyperParameters(void) const{

	return correlationFunction.getHyperParameters();

}

void  TGEKModel::setHyperParameters(vec parameters){

	assert(parameters.size() == data.getDimension());
	correlationFunction.setHyperParameters(parameters);

}


void TGEKModel::updateAuxilliaryFields(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	output.printMessage("Updating auxiliary model variables...");

	auxiliaryModel.updateAuxilliaryFields();

	if(numberOfDifferentiatedBasisFunctions > 0){

		findIndicesOfDifferentiatedBasisFunctionLocations();

	}

	generateWeightingMatrix();

	calculatePhiMatrix();



	unsigned int N = data.getNumberOfSamples();

	output.printMessage("Number of samples = ", N);

	vec ys = data.getOutputVector();
	double sumys = sum(ys);
	beta0 = sumys/N;

	output.printMessage("beta0 = ", beta0);

	linearSystemCorrelationMatrixSVD.setMatrix(WPhi);
	/* SVD decomposition R = U Sigma VT */
	linearSystemCorrelationMatrixSVD.factorize();

	generateRhsForRBFs();

	double thresholdValue = 10E-12;
	linearSystemCorrelationMatrixSVD.setThresholdForSingularValues(thresholdValue);

	w = linearSystemCorrelationMatrixSVD.solveLinearSystem(Wydot);

}

mat TGEKModel::getWeightMatrix(void) const{
	return weightMatrix;
}

vec TGEKModel::getSampleWeightsVector(void) const{
	return sampleWeights;
}

/* This function generates weights for each sample according to their functional value */
void TGEKModel::generateSampleWeights(void){

	unsigned int N = data.getNumberOfSamples();
	sampleWeights = zeros<vec>(N);
	vec y = data.getOutputVector();

	double maxy = max(y);
	double miny = min(y);
	double deltay = maxy-miny;
	for(unsigned int i=0; i<N; i++){

		y(i) = (y(i) - miny)/deltay;

	}

	vec z(N,fill::zeros);
	double mu = mean(y);
	double sigma = stddev(y);
	for(unsigned int i=0; i<N; i++){
		z(i) = (y(i) - mu)/sigma;
	}

	double b = 0.5;
	double a = 0.5/min(z);

	for(unsigned int i=0; i<N; i++){
		sampleWeights(i) = a*z(i) + b;
		if(sampleWeights(i)< 0.1) {
			sampleWeights(i) = 0.1;
		}
	}

}


void TGEKModel::generateWeightingMatrix(void){

	generateSampleWeights();


	unsigned int N = data.getNumberOfSamples();
	weightMatrix = zeros<mat>(2*N, 2*N);

	for(unsigned int i=0; i<N; i++){
		weightMatrix(i,i) = sampleWeights(i);
	}
	for(unsigned int i=N; i<2*N; i++){
		weightMatrix(i,i) = sampleWeights(i-N)*0.5;
	}

}


void TGEKModel::generateRhsForRBFs(void){

	unsigned int N = data.getNumberOfSamples();
	ydot = zeros<vec>(2*N);

	vec derivatives = data.getDirectionalDerivativesVector();
	vec y           = data.getOutputVector();

	for(unsigned int i=0; i<N; i++){
		ydot(i) = y(i) - beta0;
	}


	for(unsigned int i=0; i<N; i++){
		ydot(N+i) = derivatives(i);
	}

	/* Multiply with the weight matrix 2Nx2N * 2N */

	Wydot = weightMatrix*ydot;
}


void TGEKModel::train(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(ifInitialized);

	trainTheta();
	updateAuxilliaryFields();

	ifModelTrainingIsDone = true;
}


void TGEKModel::prepareTrainingDataForTheKrigingModel(void){

	assert(ifDataIsRead);
	mat rawData = data.getRawData();
	unsigned int N = data.getNumberOfSamples();
	unsigned int dim = data.getDimension();

	mat inputX = data.getInputMatrix();
	vec y = data.getOutputVector();

	mat trainingDataForAuxModel(N,dim+1);

	for(unsigned int i=0; i<dim; i++){

		trainingDataForAuxModel.col(i) = inputX.col(i);
	}

	trainingDataForAuxModel.col(dim) = y;

	//	trainingDataForAuxModel.print("trainingDataForAuxModel");

	saveMatToCVSFile(trainingDataForAuxModel, filenameTrainingDataAuxModel);

}

void TGEKModel::trainTheta(void){

	auxiliaryModel.train();

	vec hyperparameters = auxiliaryModel.getHyperParameters();
	unsigned int dim = data.getDimension();
	vec theta = hyperparameters.head(dim);

	//	theta.print("theta");
	correlationFunction.setHyperParameters(theta);

	remove(filenameTrainingDataAuxModel.c_str());

}

double TGEKModel::interpolate(rowvec x) const{


	unsigned int N = data.getNumberOfSamples();
	unsigned int Nd = numberOfDifferentiatedBasisFunctions;

	double sum = 0.0;
	for(unsigned int i=0; i<N; i++){
		rowvec xi = data.getRowX(i);
		double r = correlationFunction.computeCorrelation(xi,x);
		sum += w(i)*r;
	}

	for(unsigned int i=0; i<Nd; i++){

		unsigned int index = indicesDifferentiatedBasisFunctions(i);
		rowvec xi = data.getRowX(index);
		rowvec diffDirection = data.getRowDifferentiationDirection(index);
		sum +=w(i+N)*correlationFunction.computeCorrelationDot(xi, x, diffDirection);
	}


	return sum + beta0;

}
void TGEKModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const {

	auxiliaryModel.interpolateWithVariance(xp, f_tilde, ssqr);
	*f_tilde = interpolate(xp);

}
//void TGEKModel::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{
//
//	double ftilde = 0.0;
//	double ssqr   = 0.0;
//
//	interpolateWithVariance(designCalculated.dv,&ftilde,&ssqr);
//
//#if 0
//	printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
//#endif
//
//	double	sigma = sqrt(ssqr)	;
//
//#if 0
//	printf("standart_ERROR = %15.10f\n",sigma);
//#endif
//
//	double expectedImprovementValue = 0.0;
//
//	if(fabs(sigma) > EPSILON){
//
//		double ymin = data.getMinimumOutputVector();
//		double ymax = data.getMaximumOutputVector();
//
//		double improvement = 0.0;
//		improvement = ymin - ftilde;
//
//		double	Z = (improvement)/sigma;
//#if 0
//		printf("Z = %15.10f\n",Z);
//		printf("ymin = %15.10f\n",ymin);
//#endif
//
//
//		expectedImprovementValue = improvement*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
//	}
//	else{
//
//		expectedImprovementValue = 0.0;
//
//	}
//#if 0
//	printf("expectedImprovementValue = %20.20f\n",expectedImprovementValue);
//#endif
//
//	designCalculated.objectiveFunctionValue = ftilde;
//	designCalculated.valueExpectedImprovement = expectedImprovementValue;
//
//}

void TGEKModel::resetDataObjects(void){

	Phi.reset();
	WPhi.reset();
	Wydot.reset();
	beta0 = 0.0;
	auxiliaryModel.resetDataObjects();


}

void TGEKModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();

}


void TGEKModel::addNewSampleToData(rowvec newsample){

	unsigned int dim = data.getDimension();
	assert(newsample.size() == 2*dim+2);
	Bounds boxConstraints = data.getBoxConstraints();

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	rowvec x = newsample.head(dim);
	x = normalizeRowVector(x, lb, ub);

	mat inputData = data.getInputMatrix();

	bool flagTooClose= checkifTooCLose(x, inputData);

	if(!flagTooClose){

		appendRowVectorToCSVData(newsample, filenameDataInput);
		updateModelWithNewData();

		rowvec xForKriging = newsample.head(dim+1);
		auxiliaryModel.addNewSampleToData(xForKriging);

	}


}
void TGEKModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(false);

}


void TGEKModel::findIndicesOfDifferentiatedBasisFunctionLocations(void){

	assert(ifDataIsRead);
	output.printMessage("Finding the indices of differentiated basis functions...");
	output.printMessage("Number of differentiated basis functions = ",numberOfDifferentiatedBasisFunctions);
	vec y = data.getOutputVector();
	indicesDifferentiatedBasisFunctions = findIndicesKMin(y, numberOfDifferentiatedBasisFunctions);

	//	indicesDifferentiatedBasisFunctions.print();
	//	y.print();

}

void TGEKModel::initializeHyperParameters(void){

	assert(ifDataIsRead);
	unsigned int dim = data.getDimension();
	correlationFunction.setDimension(dim);
	correlationFunction.initialize();

}

void TGEKModel::calculatePhiEntriesForFunctionValues(void) {
	unsigned int N = data.getNumberOfSamples();
	/* first N equations are functional values */
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			Phi(i, j) = correlationFunction.computeCorrelation(j, i);
		}
		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;
				j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions(j);
			rowvec d = data.getRowDifferentiationDirection(index);
			Phi(i, N + j) = correlationFunction.computeCorrelationDot(index, i,
					d);
		}
	}
}

void TGEKModel::calculatePhiEntriesForDerivatives(void) {
	unsigned int N = data.getNumberOfSamples();
	/* last N equations are derivatives */
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			rowvec d = data.getRowDifferentiationDirection(i);
			Phi(N + i, j) = correlationFunction.computeCorrelationDot(j, i, d);
		}
		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;
				j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions(j);
			rowvec d1 = data.getRowDifferentiationDirection(index);
			rowvec d2 = data.getRowDifferentiationDirection(i);
			Phi(N + i, N + j) = correlationFunction.computeCorrelationDotDot(
					index, i, d1, d2);
		}
	}
}

/*
 *
 *
 *  ftilde(x) = phi1(x) * w1 + phi2(x) * w2 + ... + phiN(x) * wN +
 *
 *              dphi1(x) *wN+1 + dphi2(x) *wN+2 + ... + dphiN(x) *wN+Nd
 *
 *
 */

void TGEKModel::calculatePhiMatrix(void){

	assert(ifDataIsRead);
	assert(ifNormalized);


	correlationFunction.setInputSampleMatrix(data.getInputMatrix());

	unsigned int N = data.getNumberOfSamples();

	Phi = zeros<mat>(2*N, N + numberOfDifferentiatedBasisFunctions);

	/* first N equations are functional values */
	calculatePhiEntriesForFunctionValues();

	/* last N equations are derivatives */
	calculatePhiEntriesForDerivatives();

	/* multiply with the weight matrix */

	WPhi = weightMatrix*Phi;

}

bool TGEKModel::checkPhiMatrix(void){

	unsigned int N = data.getNumberOfSamples();

	for(unsigned int i=0; i<N; i++){

		rowvec xi = data.getRowX(i);
		double ftilde = interpolate(xi);

		double sum = 0.0;
		for(unsigned int j=0; j<N+numberOfDifferentiatedBasisFunctions;j++){

			sum +=Phi(i,j)*w(j);

		}

		double error = fabs(sum - ftilde);
		if(error > 10E-05) return false;



	}

	double epsilon = 0.00001;
	for(unsigned int i=0; i<N; i++){

		rowvec xi = data.getRowX(i);
		rowvec d = data.getRowDifferentiationDirection(i);
		rowvec xiPerturbed = xi + epsilon*d;
		double ftilde = interpolate(xi);
		double ftildePerturbed = interpolate(xiPerturbed);

		double fdValue = (ftildePerturbed - ftilde)/epsilon;

		double sum = 0.0;
		for(unsigned int j=0; j<N+numberOfDifferentiatedBasisFunctions;j++){

			sum +=Phi(N+i,j)*w(j);

		}

		double error = fabs(sum - fdValue);
		if(error > 10E-05) return false;

	}
	return true;

}
