/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
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

void TGEKModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
	auxiliaryModel.setBoxConstraints(boxConstraintsInput);

}

void TGEKModel::setDimension(unsigned int dim){

	dimension = dim;
	auxiliaryModel.setDimension(dim);
}


void TGEKModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.setDirectionalDerivativesOn();
	data.readData(filenameDataInput);

	numberOfSamples = data.getNumberOfSamples();

	ifDataIsRead = true;

	prepareTrainingDataForTheKrigingModel();
	auxiliaryModel.readData();
}

void TGEKModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalize();
	ifNormalized = true;

	auxiliaryModel.normalizeData();
}


void TGEKModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;
	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);

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

		output.printMessage("Initializing the correlation function...");
		correlationFunction.initialize();
		ifCorrelationFunctionIsInitialized = true;
	}

	numberOfHyperParameters = dim;
	w = ones<vec>(data.getNumberOfSamples() + numberOfDifferentiatedBasisFunctions);
	initializeHyperParameters();


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


void TGEKModel::calculateIndicesOfSamplesWithActiveDerivatives(void){

	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	assert(dimension>0);

	vec derivatives = data.getDirectionalDerivativesVector();


	for(unsigned int i=0; i<numberOfSamples; i++){

		bool isVerySmall = false;

		if(fabs(derivatives(i)) < EPSILON) isVerySmall = true;

		if(!isVerySmall){
			indicesOfSamplesWithActiveDerivatives.push_back(i);
		}
	}

	unsigned int howManyActiveSamples = indicesOfSamplesWithActiveDerivatives.size();

	output.printMessage("Number of samples with active tangents = ",howManyActiveSamples);

	if(numberOfDifferentiatedBasisFunctions > howManyActiveSamples){

		output.printMessage("Number of differentiated basis functions is greater than number of samples with active gradients!");
		output.printMessage("Number of differentiated basis functions is reduced to ", howManyActiveSamples);

		numberOfDifferentiatedBasisFunctions = howManyActiveSamples;
	}

	string msg = "Indices of samples with active tangents";
	output.printList(indicesOfSamplesWithActiveDerivatives, msg);

	ifActiveDeritiveSampleIndicesAreCalculated = true;


}

void TGEKModel::assembleLinearSystem(void){

	assert(ifDataIsRead);
	assert(ifNormalized);


	calculateIndicesOfSamplesWithActiveDerivatives();

	if(numberOfDifferentiatedBasisFunctions > 0){

		findIndicesOfDifferentiatedBasisFunctionLocations();

	}

	generateWeightingMatrix();
	calculatePhiMatrix();
	calculateBeta0();
	generateRhsForRBFs();

}


void TGEKModel::updateAuxilliaryFields(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	output.printMessage("Updating auxiliary model variables...");

	auxiliaryModel.updateAuxilliaryFields();


	assembleLinearSystem();

	solveLinearSystem();


}


void TGEKModel::solveLinearSystem() {

	assert(Phi.n_rows == numberOfSamples + indicesOfSamplesWithActiveDerivatives.size());
	assert(Phi.n_cols == numberOfSamples + numberOfDifferentiatedBasisFunctions);
	assert(ydot.size() == Phi.n_rows);



	linearSystemCorrelationMatrixSVD.setMatrix(Phi);
	/* SVD decomposition R = U Sigma VT */
	linearSystemCorrelationMatrixSVD.factorize();
	linearSystemCorrelationMatrixSVD.setThresholdForSingularValues(sigmaThresholdValueForSVD);
	linearSystemCorrelationMatrixSVD.setRhs(ydot);
	w = linearSystemCorrelationMatrixSVD.solveLinearSystem();


	assert(w.size() == numberOfSamples + numberOfDifferentiatedBasisFunctions);
}




mat TGEKModel::getWeightMatrix(void) const{
	return weightMatrix;
}

vec TGEKModel::getSampleWeightsVector(void) const{
	return sampleWeights;
}

/* This function generates weights for each sample according to their functional value */
void TGEKModel::generateSampleWeights(void){

	unsigned int N = numberOfSamples;
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

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);

	generateSampleWeights();

	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();

	unsigned int sizeOfWeightMatrix = numberOfSamples +  howManySamplesHaveDerivatives;


	weightMatrix = zeros<mat>(sizeOfWeightMatrix, sizeOfWeightMatrix);


	/* weights for functional values */
	for(unsigned int i=0; i<numberOfSamples; i++){
		weightMatrix(i,i) = sampleWeights(i);
	}

	double weightFactorForDerivatives = 0.5;

	for(unsigned int i=0; i<howManySamplesHaveDerivatives; i++){
		unsigned int indx = indicesOfSamplesWithActiveDerivatives.at(i);
		double weight = sampleWeights(indx);
		weightMatrix(i+numberOfSamples,i+numberOfSamples) = weight* weightFactorForDerivatives;
	}
}


void TGEKModel::generateRhsForRBFs(void){

	assert(numberOfSamples>0);

	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int N    = numberOfSamples;

	unsigned int sizeOfRhs = N + Ndot;

	ydot = zeros<vec>(sizeOfRhs);


	vec derivatives = data.getDirectionalDerivativesVector();
	vec y           = data.getOutputVector();

	for(unsigned int i=0; i<N; i++){
		ydot(i) = y(i) - beta0;
	}

	for(unsigned int i=0; i<Ndot; i++){
		unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];

		double directionalDerivative = derivatives(indx);
		ydot(i+N) = directionalDerivative;
	}

	if(ifVaryingSampleWeights){

		ydot = weightMatrix*ydot;
	}

}


void TGEKModel::calculateBeta0(void) {

	output.printMessage("Number of samples = ", numberOfSamples);
	vec ys = data.getOutputVector();
	double sumys = sum(ys);
	beta0 = sumys / numberOfSamples;
	output.printMessage("beta0 = ", beta0);
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
	unsigned int N = numberOfSamples;

	vec y = rawData.col(dimension);
	mat X = rawData.submat(0, 0, N-1,dimension-1);


	mat trainingDataForAuxModel(N,dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		trainingDataForAuxModel.col(i) = X.col(i);
	}

	trainingDataForAuxModel.col(dimension) = y;

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


	unsigned int N  = numberOfSamples;
	unsigned int Nd = numberOfDifferentiatedBasisFunctions;

	double sum = 0.0;
	for(unsigned int i=0; i<N; i++){
		rowvec xi = data.getRowX(i);
		double r = correlationFunction.computeCorrelation(xi,x);
		sum += w(i)*r;
	}

	for(unsigned int i=0; i<Nd; i++){

		unsigned int index = indicesDifferentiatedBasisFunctions[i];
		rowvec xi = data.getRowX(index);
		rowvec diffDirection = data.getRowDifferentiationDirection(index);
		sum +=w(i+N)*correlationFunction.computeCorrelationDot(xi, x, diffDirection);
	}


	return sum + beta0;

}
void TGEKModel::interpolateWithVariance(rowvec xp,double *fTilde, double *ssqr) const {

	auxiliaryModel.interpolateWithVariance(xp, fTilde, ssqr);
	*fTilde = interpolate(xp);
}


void TGEKModel::resetDataObjects(void){

	Phi.reset();
	ydot.reset();
	beta0 = 0.0;
	auxiliaryModel.resetDataObjects();


}

void TGEKModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();
	updateAuxilliaryFields();

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
	}


}
void TGEKModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(false);

}


void TGEKModel::setValuesForFindingDifferentiatedBasisIndex(vec &values) {

	vec y = data.getOutputVector();

	if (ifTargetForDifferentiatedBasisIsSet) {
		for (unsigned int i = 0; i < numberOfSamples; i++) {
			values(i) = fabs(y(i) - targetForDifferentiatedBasis);
		}
	} else {
		values = y;
	}
	for (unsigned int i = 0; i < numberOfSamples; i++) {
		if (!isIntheList(indicesOfSamplesWithActiveDerivatives, i)) {
			values(i) = LARGE;
		}
	}
}


void TGEKModel::findIndicesOfDifferentiatedBasisFunctionLocations(void){

	assert(ifDataIsRead);
	assert(numberOfSamples > 0);
	assert(numberOfDifferentiatedBasisFunctions < numberOfSamples);


	output.printMessage("Finding the indices of differentiated basis functions...");
	output.printMessage("Number of differentiated basis functions = ",numberOfDifferentiatedBasisFunctions);

	vec values(numberOfSamples, fill::zeros);
	setValuesForFindingDifferentiatedBasisIndex(values);
	indicesDifferentiatedBasisFunctions = returnKMinIndices(values, numberOfDifferentiatedBasisFunctions);

	string msg = "Indices of samples with differentiated basis functions = ";
	output.printList(indicesDifferentiatedBasisFunctions, msg);

	/*
	for (std::vector<int>::const_iterator i = indicesDifferentiatedBasisFunctions.begin(); i != indicesDifferentiatedBasisFunctions.end(); ++i){
		cout<<"i = "<<*i<<" "<<values(*i)<<"\n";

	}
	 */


}




//
//void TGEKModel::findIndicesOfDifferentiatedBasisFunctionLocations(void){
//
//	assert(ifDataIsRead);
//	output.printMessage("Finding the indices of differentiated basis functions...");
//	output.printMessage("Number of differentiated basis functions = ",numberOfDifferentiatedBasisFunctions);
//	vec y = data.getOutputVector();
//	indicesDifferentiatedBasisFunctions = findIndicesKMin(y, numberOfDifferentiatedBasisFunctions);
//
//	//	indicesDifferentiatedBasisFunctions.print();
//	//	y.print();
//
//}

void TGEKModel::initializeHyperParameters(void){

	assert(ifDataIsRead);
	unsigned int dim = data.getDimension();
	correlationFunction.setDimension(dim);
	correlationFunction.initialize();

}

void TGEKModel::calculatePhiEntriesForFunctionValues(void) {

	unsigned int N = numberOfSamples;
	/* first N equations are functional values */
	for (unsigned int i = 0; i < N; i++) {
		rowvec xi = data.getRowX(i);
		for (unsigned int j = 0; j < N; j++) {
			rowvec xj = data.getRowX(j);
			Phi(i, j) = correlationFunction.computeCorrelation(xj, xi);
		}
		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions[j];
			rowvec xAtindex = data.getRowX(index);
			rowvec d = data.getRowDifferentiationDirection(index);
			Phi(i, N + j) = correlationFunction.computeCorrelationDot(xAtindex, xi,d);
		}
	}
}

void TGEKModel::calculatePhiEntriesForDerivatives(void) {


	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();

	/* last "howManySamplesHaveDerivatives" equations are for directional derivatives */
	for (unsigned int i = 0; i < Ndot; i++) {

		unsigned int sampleIndex = indicesOfSamplesWithActiveDerivatives[i];
		rowvec xAtsampleIndex = data.getRowX(sampleIndex);

		rowvec directionAtSample = data.getRowDifferentiationDirection(sampleIndex);
		/* directional derivatives of primary basis functions */
		for (unsigned int j = 0; j < N; j++) {
			rowvec xj = data.getRowX(j);
			Phi(N + i, j) = correlationFunction.computeCorrelationDot(xj, xAtsampleIndex, directionAtSample);
		}

		/* directional derivatives of differentiated basis functions */

		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions[j];
			rowvec xAtindex = data.getRowX(index);
			rowvec directionBasis = data.getRowDifferentiationDirection(index);
			Phi(N + i, N + j) = correlationFunction.computeCorrelationDotDot(
					xAtindex, xAtsampleIndex, directionBasis, directionAtSample);
		}

	}

}


void TGEKModel::calculatePhiMatrix(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);

	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int howManyTotalDataPoints = N + Ndot;
	unsigned int howManyBasisFunctions = numberOfSamples + numberOfDifferentiatedBasisFunctions;


	correlationFunction.setInputSampleMatrix(data.getInputMatrix());


	Phi = zeros<mat>(howManyTotalDataPoints, howManyBasisFunctions);

	calculatePhiEntriesForFunctionValues();
	calculatePhiEntriesForDerivatives();

	//	assert(checkPhiMatrix());

	if(ifVaryingSampleWeights){
		unsigned int sizeOfWeightMatrix = howManyTotalDataPoints;
		assert(weightMatrix.n_rows == sizeOfWeightMatrix);
		Phi = weightMatrix*Phi;

	}



}



bool TGEKModel::checkPhiMatrix(void){


	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();


	double epsilon = 0.0000001;

	for(unsigned int i=0; i<N; i++){

		rowvec xi = data.getRowX(i);
		double ftilde = interpolate(xi);

		double sum = 0.0;
		for(unsigned int j=0; j<N+numberOfDifferentiatedBasisFunctions;j++){

			sum +=Phi(i,j)*w(j);

		}

//		printTwoScalars(sum, ftilde);

		double error = fabs(sum - ftilde);
		if(error > 10E-05) return false;

	}

	/* directional derivatives */


	for(unsigned int i=0; i<Ndot; i++){

		unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];

		rowvec xi = data.getRowX(indx);
		rowvec d  = data.getRowDifferentiationDirection(indx);

		rowvec xiPerturbedPlus = xi + epsilon*d;
		rowvec xiPerturbedMins = xi - epsilon*d;
		double ftildePerturbedPlus = interpolate(xiPerturbedPlus);
		double ftildePerturbedMins = interpolate(xiPerturbedMins);

		double fdValue = (ftildePerturbedPlus - ftildePerturbedMins)/(2.0*epsilon);



		double sum = 0.0;
		for(unsigned int j=0; j<N+numberOfDifferentiatedBasisFunctions;j++){
			sum +=Phi(N+i,j)*w(j);
		}

		double error = fabs(sum - fdValue);
		if(error > 10E-05) return false;

	}

	return true;

}

unsigned int TGEKModel::getNumberOfSamplesWithActiveGradients(void) const{
	return indicesOfSamplesWithActiveDerivatives.size();

}



