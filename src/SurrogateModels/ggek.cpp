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
	data.setDimension(dim);
	auxiliaryModel.setDimension(dim);
}


void GGEKModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;
	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);

}

void GGEKModel::setNameOfHyperParametersFile(string filename){


}
void GGEKModel::setNumberOfTrainingIterations(unsigned int N){

	numberOfTrainingIterations = N;
	auxiliaryModel.setNumberOfTrainingIterations(N);
	output.printMessage("Number of training iterations is set to ", numberOfTrainingIterations);

}

void GGEKModel::setThetaFactor(double value){
	assert(value>=1.0);
	thetaFactor = value;
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

		differentiatedCorrelationFunction.setDimension(dimension);
		differentiatedCorrelationFunction.initialize();

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
	weights = ones<vec>(numberOfSamples + indicesOfSamplesWithActiveDerivatives.size());

	auxiliaryModel.setDimension(dimension);
	auxiliaryModel.initializeSurrogateModel();

	updateAuxilliaryFields();


	ifInitialized = true;



}
void GGEKModel::printSurrogateModel(void) const{

	cout<<"GGEK Model = \n";
	cout<<"Name = "<<name<<"\n";
	cout<<"Dimension = "<<dimension<<"\n";
	cout<<"Number of samples  = "<<numberOfSamples<<"\n";

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

void GGEKModel::updateCorrelationFunctions(void){

	assert(dimension>0);
	assert(theta.size() == dimension);
	assert(gamma.size() == dimension);

	vec hyperParametersForPrimalBasis = joinVectors(theta, gamma);
	assert(hyperParametersForPrimalBasis.size() == 2*dimension);

	correlationFunction.setHyperParameters(hyperParametersForPrimalBasis);
	vec thetaForDualBasis = thetaFactor * theta;
	assert(thetaForDualBasis.size() == dimension);
	differentiatedCorrelationFunction.setHyperParameters(thetaForDualBasis);

}


void GGEKModel::setHyperParameters(vec parameters){

	assert(parameters.size() == 2*dimension);
	theta = parameters.head(dimension);
	gamma = parameters.tail(dimension);

	correlationFunction.setHyperParameters(parameters);

	differentiatedCorrelationFunction.setHyperParameters(thetaFactor*theta);


}



void GGEKModel::assembleLinearSystem(void){

	assert(ifDataIsRead);
	assert(ifNormalized);

	calculateIndicesOfSamplesWithActiveDerivatives();

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

	assert(numberOfSamples > 0);
	assert(ydot.size() == Phi.n_rows);


	SVDSystem  linearSystemCorrelationMatrixSVD;

	linearSystemCorrelationMatrixSVD.setMatrix(Phi);
	/* SVD decomposition R = U Sigma VT */
	linearSystemCorrelationMatrixSVD.factorize();
	linearSystemCorrelationMatrixSVD.setThresholdForSingularValues(sigmaThresholdValueForSVD);
	linearSystemCorrelationMatrixSVD.setRhs(ydot);

	weights = linearSystemCorrelationMatrixSVD.solveLinearSystem();

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
	determineThetaCoefficientForDualBasis();

	updateAuxilliaryFields();

	ifModelTrainingIsDone = true;

}


void GGEKModel::trainTheta(void){

	assert(dimension>0);

	auxiliaryModel.train();

	vec hyperparameters = auxiliaryModel.getHyperParameters();
	assert(hyperparameters.size() == 2*dimension);

	theta = hyperparameters.head(dimension);
	gamma = hyperparameters.tail(dimension);

	/*
	theta.print("theta");
	gamma.print("gamma");
	 */

	correlationFunction.setHyperParameters(hyperparameters);


	vec thetaForDualBasis = thetaFactor*theta;
	differentiatedCorrelationFunction.setHyperParameters(thetaForDualBasis);


}

void GGEKModel::determineThetaCoefficientForDualBasis(void){

	assert(ifDataIsRead);
	assert(dimension>0);
	assert(theta.size() == dimension);
	assert(gamma.size() == dimension);
	assert(boxConstraints.areBoundsSet());

	output.printMessage("Estimation for the theta amplification factor...");

	GGEKModel auxiliaryModelForThetaCoefficient;
	auxiliaryModelForThetaCoefficient.setDimension(dimension);
	auxiliaryModelForThetaCoefficient.setBoxConstraints(boxConstraints);

	mat rawData = data.getRawData();

	unsigned int howManyTrainingSamplesToUse = numberOfSamples/2;
	assert(howManyTrainingSamplesToUse>0);

	rawData = shuffleRows(rawData);


	mat halfData = rawData.submat(0, 0, howManyTrainingSamplesToUse -1, rawData.n_cols-1);
	halfData.save("trainingDataForTheThetaAuxModel.csv", csv_ascii);

	mat secondhalfData = rawData.submat(howManyTrainingSamplesToUse, 0, numberOfSamples-1, dimension);
	secondhalfData.save("testDataForTheThetaAuxModel.csv", csv_ascii);

	assert(halfData.n_cols == 2*dimension+1);
	assert(secondhalfData.n_cols == dimension+1);
	assert(halfData.n_rows + secondhalfData.n_rows  == numberOfSamples);

	auxiliaryModelForThetaCoefficient.setNameOfInputFile("trainingDataForTheThetaAuxModel.csv");
	auxiliaryModelForThetaCoefficient.setNameOfInputFileTest("testDataForTheThetaAuxModel.csv");

	auxiliaryModelForThetaCoefficient.readData();
	auxiliaryModelForThetaCoefficient.normalizeData();

	auxiliaryModelForThetaCoefficient.readDataTest();
	auxiliaryModelForThetaCoefficient.normalizeDataTest();

	auxiliaryModelForThetaCoefficient.initializeSurrogateModel();
	auxiliaryModelForThetaCoefficient.assembleLinearSystem();


	vec hyperParameters = joinVectors(theta,gamma);
	assert(hyperParameters.size() == 2*dimension);

	double exponentStart = 0.0;
	double exponentEnd   = 3.0;
	double deltaExponent = (exponentEnd-exponentStart)/ numberOfIterationsToDetermineThetaFactor;

	double bestMSE = LARGE;
	double bestFactor = 1.0;


	double exponent = exponentStart;

	for(unsigned int i=0; i<numberOfIterationsToDetermineThetaFactor; i++){


		double valueToTry = pow(10.0,exponent);

		auxiliaryModelForThetaCoefficient.setThetaFactor(valueToTry);
		auxiliaryModelForThetaCoefficient.setHyperParameters(hyperParameters);

		auxiliaryModelForThetaCoefficient.calculatePhiMatrix();
		auxiliaryModelForThetaCoefficient.solveLinearSystem();
		auxiliaryModelForThetaCoefficient.tryOnTestData();

		double MSE = auxiliaryModelForThetaCoefficient.generalizationError;
		assert(MSE > 0.0);

		if(MSE < bestMSE){

			bestMSE = MSE;
			bestFactor = valueToTry;
		}

		exponent +=deltaExponent;
	}

	//	printTwoScalars(bestMSE, bestFactor);

	thetaFactor = bestFactor;

	output.printMessage("optimized theta factor = ", thetaFactor);

	updateCorrelationFunctions();

	ifThetaFactorOptimizationIsDone = true;

	remove("trainingDataForTheThetaAuxModel.csv");
	remove("testDataForTheThetaAuxModel.csv");

}


double GGEKModel::interpolate(rowvec x) const{

	unsigned int N  = numberOfSamples;
	unsigned int Nd = indicesOfSamplesWithActiveDerivatives.size();

	mat gradientsMatrix = data.getGradientMatrix();
	vec r(N+Nd);


	double sum = 0.0;
	for(unsigned int i=0; i<N; i++){
		rowvec xi = data.getRowX(i);
		r(i) = correlationFunction.computeCorrelation(xi,x);
		sum += weights(i)*r(i);
	}

	for(unsigned int i=0; i<Nd; i++){

		unsigned int index = indicesOfSamplesWithActiveDerivatives[i];
		rowvec xi = data.getRowX(index);
		rowvec grad = gradientsMatrix.row(index);
		rowvec diffDirection = makeUnitVector(grad);

		r(N+i) = differentiatedCorrelationFunction.computeCorrelationDot(xi, x, diffDirection);
		sum +=weights(N+i)*r(N+i);
	}

	return sum + beta0;
}



void GGEKModel::interpolateWithVariance(rowvec xp,double *fTilde,double *ssqr) const{

	auxiliaryModel.interpolateWithVariance(xp, fTilde, ssqr);
	*fTilde = interpolate(xp);


}


void GGEKModel::setTargetForSampleWeights(double value){
	targetForSampleWeights = value;
	ifTargetForSampleWeightsIsSet= true;
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

	string msg = "Indices of samples with active gradients";
	output.printList(indicesOfSamplesWithActiveDerivatives, msg);

	ifActiveDeritiveSampleIndicesAreCalculated = true;
}


void GGEKModel::generateWeightingMatrix(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);


	if(ifVaryingSampleWeights){
		generateSampleWeights();


		unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();

		unsigned int sizeOfWeightMatrix = numberOfSamples +  howManySamplesHaveDerivatives;

		weightMatrix.reset();
		weightMatrix = zeros<mat>(sizeOfWeightMatrix, sizeOfWeightMatrix);


		/* weights for functional values */
		for(unsigned int i=0; i<numberOfSamples; i++){
			weightMatrix(i,i) = sampleWeights(i);
		}


		for(unsigned int i=0; i<howManySamplesHaveDerivatives; i++){
			unsigned int indx = indicesOfSamplesWithActiveDerivatives.at(i);
			double weight = sampleWeights(indx);
			weightMatrix(i+numberOfSamples,i+numberOfSamples) = weight* weightFactorForDerivatives;
		}

	}
}

void GGEKModel::generateRhsForRBFs(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int N    = numberOfSamples;

	unsigned int sizeOfRhs = N + Ndot;

	ydot.reset();
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
		rowvec d    = makeUnitVector(grad);

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
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	mat gradients = data.getGradientMatrix();

	/* first N equations are functional values */
	for (unsigned int i = 0; i < N; i++) {

		rowvec xi = data.getRowX(i);

		for (unsigned int j = 0; j < N; j++) {
			rowvec xj = data.getRowX(j);
			Phi(i, j) = correlationFunction.computeCorrelation(xj, xi);
		}


		for (unsigned int j = 0; j < Ndot;j++) {
			unsigned int index = indicesOfSamplesWithActiveDerivatives[j];
			rowvec xAtIndex = data.getRowX(index);
			rowvec g = gradients.row(index);
			rowvec d = makeUnitVector(g);

			Phi(i, N + j) = differentiatedCorrelationFunction.computeCorrelationDot(xAtIndex, xi,d);
		}


	}


}

void GGEKModel::calculatePhiEntriesForDerivatives(void) {

	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();

	mat gradients = data.getGradientMatrix();

	/* last "howManySamplesHaveDerivatives" equations are for directional derivatives */
	for (unsigned int i = 0; i < Ndot; i++) {



		unsigned int sampleIndex = indicesOfSamplesWithActiveDerivatives[i];
		rowvec xAtsampleIndex = data.getRowX(sampleIndex);

		rowvec grad = gradients.row(sampleIndex);
		rowvec directionAtSample = makeUnitVector(grad);

		/* directional derivatives of primary basis functions */
		for (unsigned int j = 0; j < N; j++) {
			rowvec xj = data.getRowX(j);
			Phi(N + i, j) = correlationFunction.computeCorrelationDot(xj, xAtsampleIndex, directionAtSample);
		}

		/* directional derivatives of differentiated basis functions */

		for (unsigned int j = 0; j < Ndot;j++) {
			unsigned int index = indicesOfSamplesWithActiveDerivatives[j];
			rowvec xAtindex = data.getRowX(index);

			rowvec g = gradients.row(index);
			rowvec directionBasis = makeUnitVector(g);

			Phi(N + i, N + j) = differentiatedCorrelationFunction.computeCorrelationDotDot(
					xAtindex, xAtsampleIndex, directionBasis, directionAtSample);
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
	unsigned int howManyBasisFunctions = numberOfSamples + howManySamplesHaveDerivatives;


	Phi.reset();
	Phi = zeros<mat>(howManyTotalDataPoints, howManyBasisFunctions);

	calculatePhiEntriesForFunctionValues();
	calculatePhiEntriesForDerivatives();


	if(ifVaryingSampleWeights){

		unsigned int sizeOfWeightMatrix = howManyTotalDataPoints;
		assert(weightMatrix.n_rows == sizeOfWeightMatrix);
		Phi = weightMatrix*Phi;
	}

}

bool GGEKModel::checkResidual(void) const{

	vec residual = ydot - Phi*weights;

//	residual.print("residual");

	if(norm(residual) < 10E-3) return true;

	else return false;
}


bool GGEKModel::checkPhiMatrix(void){


	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();

	assert(weights.size() == N + Ndot);

	double epsilon = 0.0000001;


	for(unsigned int i=0; i<N; i++){

		rowvec xi = data.getRowX(i);

		double ftilde = interpolate(xi);

		double sum = 0.0;
		for(unsigned int j=0; j<N+Ndot;j++){

			sum +=Phi(i,j)*weights(j);

		}

		sum+=beta0;

//		printTwoScalars(sum, ftilde);
		double error = fabs(sum - ftilde);
		if(error > 10E-05) return false;

	}

	/* directional derivatives */


	mat gradients = data.getGradientMatrix();


	for(unsigned int i=0; i<Ndot; i++){

		unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];

		rowvec xi = data.getRowX(indx);
		rowvec g = gradients.row(indx);
		rowvec d = makeUnitVector(g);

		rowvec xiPerturbedPlus = xi + epsilon*d;
		rowvec xiPerturbedMins = xi - epsilon*d;
		double ftildePerturbedPlus = interpolate(xiPerturbedPlus);
		double ftildePerturbedMins = interpolate(xiPerturbedMins);

		double fdValue = (ftildePerturbedPlus - ftildePerturbedMins)/(2.0*epsilon);
		double sum = 0.0;
		for(unsigned int j=0; j<N+Ndot;j++){
			sum +=Phi(N+i,j)*weights(j);
		}

		double error = fabs(sum - fdValue);
		if(error > 10E-05) return false;

	}

	return true;
}

void GGEKModel::resetDataObjects(void){

	Phi.reset();
	weights.reset();
	weightMatrix.reset();
	indicesOfSamplesWithActiveDerivatives.clear();
	beta0 = 0.0;
	auxiliaryModel.resetDataObjects();


}

void GGEKModel::addNewSampleToData(rowvec newsample){

	assert(newsample.size() > 0);

	rowvec sampleToAdd(2*dimension+1, fill::zeros);
	copyRowVector(sampleToAdd, newsample);

	Bounds boxConstraints = data.getBoxConstraints();

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	rowvec x = sampleToAdd.head(dimension);
	x = normalizeRowVector(x, lb, ub);

	mat inputData = data.getInputMatrix();


	bool flagTooClose= checkifTooCLose(x, inputData);


	if(!flagTooClose){

		appendRowVectorToCSVData(sampleToAdd, filenameDataInput);
		updateModelWithNewData();
	}


}


void GGEKModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(false);

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
