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
#include "test_functions.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;
using namespace std;


void GGEKModel::setName(string label){

	assert(isNotEmpty(label));
	name = label;
	auxiliaryModel.setName(label);

	filenameTrainingDataAuxModel = name + "_aux.csv";

}


void GGEKModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
	auxiliaryModel.setBoxConstraints(boxConstraintsInput);
}

void GGEKModel::setDimension(unsigned int dim){

	dimension = dim;
	auxiliaryModel.setDimension(dim);
}


void GGEKModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;
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

	vec hyperParameters = correlationFunction.getHyperParameters();
	hyperParameters.print("theta = ");

}
void GGEKModel::saveHyperParameters(void) const{


}
void GGEKModel::loadHyperParameters(void){


}

void GGEKModel::assembleLinearSystem(void){

	assert(ifDataIsRead);
	assert(ifNormalized);

	calculateUnitGradientVectors();
	calculateIndicesOfSamplesWithActiveDerivatives();

	if(numberOfDifferentiatedBasisFunctions > 0){
		findIndicesOfDifferentiatedBasisFunctionLocations();
	}


	generateWeightingMatrix();
	calculatePhiMatrix();

	calculateBeta0();
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

	assert(ydot.size() == Phi.n_rows);



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

	//	theta.print("theta");

	correlationFunction.setHyperParameters(theta);


}


double GGEKModel::interpolate(rowvec x) const{


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
		rowvec diffDirection = unitGradientsMatrix.row(index);
		sum +=w(i+N)*correlationFunction.computeCorrelationDot(xi, x, diffDirection);
	}


	return sum + beta0;
}



void GGEKModel::interpolateWithVariance(rowvec xp,double *fTilde,double *ssqr) const{

	auxiliaryModel.interpolateWithVariance(xp, fTilde, ssqr);
	*fTilde = interpolate(xp);


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




void GGEKModel::calculateUnitGradientVectors(void){

	assert(ifDataIsRead);
	assert(dimension>0);

	unitGradientsMatrix = zeros<mat>(numberOfSamples, dimension);
	mat gradient = data.getGradientMatrix();

	for(unsigned int i=0; i<numberOfSamples; i++){
		rowvec grad =  gradient.row(i);
		unitGradientsMatrix.row(i) = makeUnitVector(grad);

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

	assert(ifDataIsRead);
	assert(numberOfSamples);

	sampleWeights = zeros<vec>(numberOfSamples);
	vec y = data.getOutputVector();
	double targetValue = min(y);

	if(ifTargetForSampleWeightsIsSet){
		targetValue = targetForSampleWeights;

	}

	vec yWeightCriteria(numberOfSamples, fill::zeros);

	for(unsigned int i=0; i<numberOfSamples; i++){
		yWeightCriteria(i) = fabs( y(i) - targetValue );
	}

	vec z(numberOfSamples,fill::zeros);
	double mu = mean(yWeightCriteria);
	double sigma = stddev(yWeightCriteria);
	for(unsigned int i=0; i<numberOfSamples; i++){
		z(i) = (yWeightCriteria(i) - mu)/sigma;
	}

	double b = 0.5;
	double a = 0.5/min(z);

	for(unsigned int i=0; i<numberOfSamples; i++){
		sampleWeights(i) = a*z(i) + b;
		if(sampleWeights(i)< 0.1) {
			sampleWeights(i) = 0.1;
		}
	}

	if(output.ifScreenDisplay){
		printSampleWeights();
	}

}


void GGEKModel::printSampleWeights(void) const{

	assert(numberOfSamples>0);
	assert(sampleWeights.size() == numberOfSamples);

	vec y = data.getOutputVector();
	for(unsigned int i=0; i<numberOfSamples; i++){
		std::cout<<"y("<<i<<") = "<<y(i)<<", w = " << sampleWeights(i) << "\n";
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

void GGEKModel::generateRhsForRBFs(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);
	assert(numberOfSamples>0);
	assert(unitGradientsMatrix.n_rows == numberOfSamples);

	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int N    = numberOfSamples;

	unsigned int sizeOfRhs = N + Ndot;

	ydot = zeros<vec>(sizeOfRhs);

	mat gradients 	= data.getGradientMatrix();
	vec y           = data.getOutputVector();

	/* first functional values */
	for(unsigned int i=0; i<N; i++){
		ydot(i) = y(i) - beta0;
	}

	/* then directional derivatives */


	for(unsigned int i=0; i<Ndot; i++){
		unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];
		rowvec grad = gradients.row(indx);
		rowvec d = unitGradientsMatrix.row(indx);

		double directionalDerivative = dot(d,grad);

		ydot(i+N) = directionalDerivative;

	}

	if(ifVaryingSampleWeights){

		assert(weightMatrix.n_rows == sizeOfRhs);
		assert(weightMatrix.n_cols == sizeOfRhs);
		ydot = weightMatrix*ydot;
	}

}

void GGEKModel::calculatePhiEntriesForFunctionValues(void) {

	unsigned int N = numberOfSamples;
	/* first N equations are functional values */
	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < N; j++) {
			Phi(i, j) = correlationFunction.computeCorrelation(j, i);
		}
		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions[j];
			rowvec d = unitGradientsMatrix.row(index);
			Phi(i, N + j) = correlationFunction.computeCorrelationDot(index, i,
					d);
		}
	}
}

void GGEKModel::calculatePhiEntriesForDerivatives(void) {

	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();

	/* last "howManySamplesHaveDerivatives" equations are for directional derivatives */
	for (unsigned int i = 0; i < Ndot; i++) {

		unsigned int sampleIndex = indicesOfSamplesWithActiveDerivatives[i];
		rowvec directionAtSample = unitGradientsMatrix.row(sampleIndex);

		/* directional derivatives of primary basis functions */
		for (unsigned int j = 0; j < N; j++) {

			Phi(N + i, j) = correlationFunction.computeCorrelationDot(j, sampleIndex, directionAtSample);
		}

		/* directional derivatives of differentiated basis functions */

		for (unsigned int j = 0; j < numberOfDifferentiatedBasisFunctions;j++) {
			unsigned int index = indicesDifferentiatedBasisFunctions[j];

			rowvec directionBasis = unitGradientsMatrix.row(index);
			Phi(N + i, N + j) = correlationFunction.computeCorrelationDotDot(
					index, sampleIndex, directionBasis, directionAtSample);
		}

	}
}


void GGEKModel::calculatePhiMatrix(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int howManyTotalDataPoints = numberOfSamples + howManySamplesHaveDerivatives;
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


bool GGEKModel::checkPhiMatrix(void){


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
		printScalar(sum);
		printScalar(ftilde);

		double error = fabs(sum - ftilde);
		if(error > 10E-05) return false;

	}

	/* directional derivatives */


	mat gradients = data.getGradientMatrix();


	for(unsigned int i=0; i<Ndot; i++){

		printScalar(i);
		unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];

		rowvec xi = data.getRowX(indx);
		rowvec d = unitGradientsMatrix.row(indx);

		rowvec xiPerturbedPlus = xi + epsilon*d;
		rowvec xiPerturbedMins = xi - epsilon*d;
		double ftildePerturbedPlus = interpolate(xiPerturbedPlus);
		double ftildePerturbedMins = interpolate(xiPerturbedMins);

		double fdValue = (ftildePerturbedPlus - ftildePerturbedMins)/(2.0*epsilon);

		printScalar(fdValue);


		double sum = 0.0;
		for(unsigned int j=0; j<N+numberOfDifferentiatedBasisFunctions;j++){
			sum +=Phi(N+i,j)*w(j);
		}

		printScalar(sum);


		double error = fabs(sum - fdValue);
		if(error > 10E-05) return false;

	}




	return true;

}




void GGEKModel::resetDataObjects(void){

	Phi.reset();
	w.reset();
	weightMatrix.reset();
	indicesDifferentiatedBasisFunctions.clear();
	indicesOfSamplesWithActiveDerivatives.clear();
	unitGradientsMatrix.reset();

	beta0 = 0.0;
	auxiliaryModel.resetDataObjects();


}

void GGEKModel::addNewSampleToData(rowvec newsample){


	assert(newsample.size() == 2*dimension+1);
	Bounds boxConstraints = data.getBoxConstraints();

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	rowvec x = newsample.head(dimension);
	x = normalizeRowVector(x, lb, ub);

	mat inputData = data.getInputMatrix();


	bool flagTooClose= checkifTooCLose(x, inputData);


	if(!flagTooClose){

		appendRowVectorToCSVData(newsample, filenameDataInput);
		updateModelWithNewData();
	}


}


void GGEKModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();
	updateAuxilliaryFields();

}


unsigned int GGEKModel::getNumberOfSamplesWithActiveGradients(void) const{
	return indicesOfSamplesWithActiveDerivatives.size();

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
	return unitGradientsMatrix;
}
