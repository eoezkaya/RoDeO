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

#include<iostream>
#include<string>
#include<cassert>


#include "ggek.hpp"
#include "auxiliary_functions.hpp"
#include "kriging_training.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;
using namespace std;


void GGEKModel::setName(string label){

	assert(isNotEmpty(label));
	name = label;
	linearModel.setName(label);
	auxiliaryModel.setName(label);

	filenameTrainingDataAuxModel = name + "_aux.csv";

}


void GGEKModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
	auxiliaryModel.setBoxConstraints(boxConstraintsInput);
	linearModel.setBoxConstraints(boxConstraintsInput);
}

void GGEKModel::setDimension(unsigned int dim){

	dimension = dim;
	auxiliaryModel.setDimension(dim);
	linearModel.setDimension(dim);

}


void GGEKModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;
	linearModel.setNameOfInputFile(filenameTrainingDataAuxModel);
	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);

}

void GGEKModel::setNameOfHyperParametersFile(string filename){


}
void GGEKModel::setNumberOfTrainingIterations(unsigned int){


}

void GGEKModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.setGradientsOn();
	data.readData(filenameDataInput);

	numberOfSamples = data.getNumberOfSamples();
	ifDataIsRead = true;

	prepareTrainingDataForTheKrigingModel();

	auxiliaryModel.readData();

	if(ifUsesLinearRegression){

		linearModel.readData();
	}

}

void GGEKModel::prepareTrainingDataForTheKrigingModel(void){

	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	assert(dimension>0);
	assert(isNotEmpty(filenameTrainingDataAuxModel));

	mat rawData = data.getRawData();

	vec y = rawData.col(dimension);
	mat X = rawData.submat(0, 0, numberOfSamples-1,dimension-1);

	mat trainingDataForAuxModel(numberOfSamples,dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		trainingDataForAuxModel.col(i) = X.col(i);
	}

	trainingDataForAuxModel.col(dimension) = y;

	saveMatToCVSFile(trainingDataForAuxModel, filenameTrainingDataAuxModel);

}




void GGEKModel::normalizeData(void){
	data.normalize();
	ifNormalized = true;

	auxiliaryModel.normalizeData();

	if(ifUsesLinearRegression){

		linearModel.normalizeData();
	}

}

void GGEKModel::initializeCorrelationFunction(void){

	assert(dimension>0);

	if(!ifCorrelationFunctionIsInitialized){
		correlationFunction.setDimension(dimension);
		correlationFunction.initialize();
		ifCorrelationFunctionIsInitialized = true;
	}


}


void GGEKModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);

	output.printMessage("Initializing the generalized gradient enhanced Kriging model...");

	mat X = data.getInputMatrix();
	correlationFunction.setInputSampleMatrix(X);
	initializeCorrelationFunction();

	numberOfHyperParameters = dimension;
	w = ones<vec>(numberOfSamples + numberOfDifferentiatedBasisFunctions);

	auxiliaryModel.initializeSurrogateModel();

	if(ifUsesLinearRegression){

		linearModel.initializeSurrogateModel();
	}

	ifInitialized = true;



}
void GGEKModel::printSurrogateModel(void) const{

	cout<<"GGEK Model = \n";
	cout<<"Name = "<<name<<"\n";
	cout<<"Dimension = "<<dimension<<"\n";
	cout<<"Number of samples  = "<<numberOfSamples<<"\n";
	cout<<"Number of differentiated basis functions used = "<<numberOfDifferentiatedBasisFunctions<<"\n";

	cout<<"Data = \n";
	data.print();

	correlationFunction.print();

	cout<<"\n\nAuxiliarly model = \n";

	auxiliaryModel.printSurrogateModel();




}
void GGEKModel::printHyperParameters(void) const{


}
void GGEKModel::saveHyperParameters(void) const{


}
void GGEKModel::loadHyperParameters(void){


}

void GGEKModel::assembleLinearSystem(void){

	assert(ifDataIsRead);
	assert(ifNormalized);


	calculateIndicesOfSamplesWithActiveDerivatives();

	if(numberOfDifferentiatedBasisFunctions > 0){

		findIndicesOfDifferentiatedBasisFunctionLocations();
		setDifferentiationDirectionsForDifferentiatedBasis();
	}

	generateWeightingMatrix();
	calculatePhiMatrix();
	generateRhsForRBFs();



}

void GGEKModel::calculateBeta0() {

	output.printMessage("Number of samples = ", numberOfSamples);
	vec ys = data.getOutputVector();
	double sumys = sum(ys);
	beta0 = sumys / numberOfSamples;
	output.printMessage("beta0 = ", beta0);
}

void GGEKModel::solveLinearSystem() {

	assert(Phi.n_rows == numberOfSamples + dimension * indicesOfSamplesWithActiveDerivatives.size());
	assert(Phi.n_cols == numberOfSamples + numberOfDifferentiatedBasisFunctions);


	linearSystemCorrelationMatrixSVD.setMatrix(Phi);
	/* SVD decomposition R = U Sigma VT */
	linearSystemCorrelationMatrixSVD.factorize();
	linearSystemCorrelationMatrixSVD.setThresholdForSingularValues(sigmaThresholdValueForSVD);
	linearSystemCorrelationMatrixSVD.setRhs(ydot);
	w = linearSystemCorrelationMatrixSVD.solveLinearSystem();
}

void GGEKModel::updateAuxilliaryFields(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	output.printMessage("Updating auxiliary model variables...");

	auxiliaryModel.updateAuxilliaryFields();

	assembleLinearSystem();

	solveLinearSystem();
}
void GGEKModel::train(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(ifInitialized);

	trainTheta();
	updateAuxilliaryFields();

	ifModelTrainingIsDone = true;

}


void GGEKModel::trainTheta(void){

	auxiliaryModel.train();

	vec hyperparameters = auxiliaryModel.getHyperParameters();
	vec theta = hyperparameters.head(dimension);

	correlationFunction.setHyperParameters(theta);


}


double GGEKModel::interpolate(rowvec x) const{

	double sum = 0.0;
	for(unsigned int i=0; i<numberOfSamples; i++){
		rowvec xi = data.getRowX(i);
		double r = correlationFunction.computeCorrelation(xi,x);
		sum += w(i)*r;
	}

	for(unsigned int i=0; i<numberOfDifferentiatedBasisFunctions; i++){
		unsigned int index = indicesDifferentiatedBasisFunctions[i];
		rowvec d = differentiationDirectionBasis.row(i);
		rowvec xi = data.getRowX(index);
		/*  differentiated basis function at xi at x */
		double r = correlationFunction.computeCorrelationDot(xi, x, d);
		sum += w(i+numberOfSamples)*r;

	}
	return sum;
}
void GGEKModel::interpolateWithVariance(rowvec xp,double *fTilde,double *ssqr) const{

	auxiliaryModel.interpolateWithVariance(xp, fTilde, ssqr);
	*fTilde = interpolate(xp);


}

void GGEKModel::addNewSampleToData(rowvec newsample){


}
void GGEKModel::addNewLowFidelitySampleToData(rowvec newsample){



}

void GGEKModel::setNumberOfDifferentiatedBasisFunctionsUsed(unsigned int n){

	assert(n<= numberOfSamples);

	numberOfDifferentiatedBasisFunctions = n;
}

void GGEKModel::setValuesForFindingDifferentiatedBasisIndex(vec &values) {

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

void GGEKModel::findIndicesOfDifferentiatedBasisFunctionLocations(void){

	assert(ifDataIsRead);
	assert(numberOfSamples > 0);
	assert(numberOfDifferentiatedBasisFunctions < numberOfSamples);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);


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




void GGEKModel::setDifferentiationDirectionsForDifferentiatedBasis(void){

	assert(ifDataIsRead);
	assert(dimension>0);
	assert(indicesDifferentiatedBasisFunctions.size() == numberOfDifferentiatedBasisFunctions);

	differentiationDirectionBasis = zeros<mat>(numberOfDifferentiatedBasisFunctions, dimension);
	mat gradient = data.getGradientMatrix();


	for(unsigned int i=0; i<numberOfDifferentiatedBasisFunctions; i++){

		unsigned int indx = indicesDifferentiatedBasisFunctions[i];
		rowvec grad =  gradient.row(indx);
		rowvec gradUnitVector = makeUnitVector(grad);

		differentiationDirectionBasis.row(i) = gradUnitVector;

	}

}


void GGEKModel::setTargetForSampleWeights(double value){
	targetForSampleWeights = value;
	ifTargetForSampleWeightsIsSet= true;
}

void GGEKModel::setTargetForDifferentiatedBasis(double value){
	targetForDifferentiatedBasis = value;
	ifTargetForDifferentiatedBasisIsSet= true;
}



/* This function generates weights for each sample according to a target value */
void GGEKModel::generateSampleWeights(void){

	sampleWeights = zeros<vec>(numberOfSamples);
	vec y = data.getOutputVector();
	double targetValue = min(y);

	if(ifTargetForSampleWeightsIsSet){
		targetValue = targetForSampleWeights;

	}

	for(unsigned int i=0; i<numberOfSamples; i++){
		y(i) = fabs(y(i) - targetValue);
	}

	vec z(numberOfSamples,fill::zeros);
	double mu = mean(y);
	double sigma = stddev(y);
	for(unsigned int i=0; i<numberOfSamples; i++){
		z(i) = (y(i) - mu)/sigma;
	}

	double b = 0.5;
	double a = 0.5/min(z);

	for(unsigned int i=0; i<numberOfSamples; i++){
		sampleWeights(i) = a*z(i) + b;
		if(sampleWeights(i)< 0.1) {
			sampleWeights(i) = 0.1;
		}
	}


}


void GGEKModel::calculateIndicesOfSamplesWithActiveDerivatives(void){

	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	assert(dimension>0);

	mat gradients = data.getGradientMatrix();

	assert(gradients.n_cols == dimension);
	assert(gradients.n_rows == numberOfSamples);

	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec gradVec = gradients.row(i);

		if(!gradVec.is_zero()){
			indicesOfSamplesWithActiveDerivatives.push_back(i);
		}
	}

	unsigned int howManyActiveSamples = indicesOfSamplesWithActiveDerivatives.size();

	output.printMessage("Number of samples with active gradients = ",howManyActiveSamples);

	if(numberOfDifferentiatedBasisFunctions > howManyActiveSamples){

		output.printMessage("Number of differentiated basis functions is greater than number of samples with active gradients!");
		output.printMessage("Number of differentiated basis functions is reduced to ", howManyActiveSamples);

		numberOfDifferentiatedBasisFunctions = howManyActiveSamples;
	}

	string msg = "Indices of samples with active gradients";
	output.printList(indicesOfSamplesWithActiveDerivatives, msg);


	ifActiveDeritiveSampleIndicesAreCalculated = true;
}


void GGEKModel::generateWeightingMatrix(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);

	generateSampleWeights();

	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();

	unsigned int sizeOfWeightMatrix = numberOfSamples + dimension * howManySamplesHaveDerivatives;

	weightMatrix = zeros<mat>(sizeOfWeightMatrix, sizeOfWeightMatrix);


	/* weights for functional values */
	for(unsigned int i=0; i<numberOfSamples; i++){
		weightMatrix(i,i) = sampleWeights(i);
	}


	for(unsigned int i=0; i<howManySamplesHaveDerivatives; i++){

		unsigned int indx = indicesOfSamplesWithActiveDerivatives.at(i);
		double weight = sampleWeights(indx);

		for(unsigned int j=0; j<dimension; j++){

			unsigned int offset = numberOfSamples + i*dimension + j;
			weightMatrix(offset,offset) = weight*0.5;
		}

	}

}

void GGEKModel::generateRhsForRBFs(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int sizeOfWeightMatrix = numberOfSamples + dimension * howManySamplesHaveDerivatives;

	ydot = zeros<vec>(sizeOfWeightMatrix);

	mat gradients 	= data.getGradientMatrix();
	vec y           = data.getOutputVector();


	/* first functional values */
	for(unsigned int i=0; i<numberOfSamples; i++){
		ydot(i) = y(i) - beta0;
	}


	/* then derivatives */

	unsigned int offset = numberOfSamples;

	for(unsigned int i=0; i<howManySamplesHaveDerivatives; i++){

		unsigned int indx = indicesOfSamplesWithActiveDerivatives.at(i);
		for(unsigned int j=0; j<dimension; j++){
			ydot(offset) = gradients(indx,j);
			offset++;
		}

	}

	if(ifVaryingSampleWeights){

		assert(weightMatrix.n_rows == sizeOfWeightMatrix);
		ydot = weightMatrix*ydot;
	}


}


void GGEKModel::calculatePhiEntriesForFunctionValues(void) {

	/* first N equations are for the functional values */
	for (unsigned int i = 0; i < numberOfSamples; i++) {

		/* primal basis functions */
		for (unsigned int j = 0; j < numberOfSamples; j++) {
			/*  jth primal basis function evaluated at x_i */
			Phi(i, j) = correlationFunction.computeCorrelation(j, i);
		}

		/* differentiated basis functions */
		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;j++) {

			unsigned int index = indicesDifferentiatedBasisFunctions[j];
			rowvec d = differentiationDirectionBasis.row(j);
			/*  differentiated basis function at the index = indexInThetrainingData evaluated at x_i */
			Phi(i, numberOfSamples + j) = correlationFunction.computeCorrelationDot(index, i, d);
		}
	}


}

void GGEKModel::calculatePhiEntriesForDerivatives(void) {

	mat unitDirections(dimension, dimension, fill::eye);

	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();

	unsigned int offset = numberOfSamples;
	for (unsigned int sampleIndex = 0; sampleIndex < howManySamplesHaveDerivatives; sampleIndex++) {

		unsigned int indexSample = indicesOfSamplesWithActiveDerivatives[sampleIndex];

		for (unsigned int direction = 0; direction < dimension; direction++) {

			rowvec d2 = unitDirections.row(direction);

			/* primal basis functions */
			for (unsigned int indexBasisFunction = 0; indexBasisFunction < numberOfSamples; indexBasisFunction++) {

				Phi(offset, indexBasisFunction) =
						correlationFunction.computeCorrelationDot(indexBasisFunction, indexSample, d2);

			}

			/* differentiated basis functions */
			for (unsigned int i = 0; i < numberOfDifferentiatedBasisFunctions;i++) {
				unsigned int indexBasisFunctionDifferentiated = indicesDifferentiatedBasisFunctions[i];
				rowvec d1 = differentiationDirectionBasis.row(i);
				Phi(offset, numberOfSamples + i) = correlationFunction.computeCorrelationDotDot(
						indexBasisFunctionDifferentiated, indexSample, d1, d2);

			}

			offset++;
		}
	}

}




void GGEKModel::calculatePhiMatrix(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int howManyDerivativeDataPoints = howManySamplesHaveDerivatives * dimension;
	unsigned int howManyDataPoints = numberOfSamples;
	unsigned int howManyTotalDataPoints = howManyDataPoints + howManyDerivativeDataPoints;
	unsigned int howManyBasisFunctions = numberOfSamples + numberOfDifferentiatedBasisFunctions;

	correlationFunction.setInputSampleMatrix(data.getInputMatrix());


	Phi = zeros<mat>(howManyTotalDataPoints, howManyBasisFunctions);

	calculatePhiEntriesForFunctionValues();
	calculatePhiEntriesForDerivatives();


	if(ifVaryingSampleWeights){
		unsigned int sizeOfWeightMatrix = numberOfSamples + howManyDerivativeDataPoints;
		assert(weightMatrix.n_rows == sizeOfWeightMatrix);
		Phi = weightMatrix*Phi;

	}


}

vector<int> GGEKModel::getIndicesOfDifferentiatedBasisFunctionLocations(void) const{
	return indicesDifferentiatedBasisFunctions;
}

mat GGEKModel::getPhiMatrix(void) const{
	return Phi;
}
mat GGEKModel::getWeightMatrix(void) const{
	return weightMatrix;
}

vec GGEKModel::getSampleWeightsVector(void) const{
	return sampleWeights;
}
mat GGEKModel::getGradient(void) const{
	assert(ifDataIsRead);
	return data.getGradientMatrix();
}

mat GGEKModel::getDifferentiationDirectionsMatrix(void) const{
	return differentiationDirectionBasis;
}
