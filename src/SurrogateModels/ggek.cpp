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


void GeneralizedDerivativeEnhancedModel::setName(string label){

	assert(isNotEmpty(label));
	name = label;
	filenameHyperparameters = name + "_hyperparameters.csv";

	auxiliaryModel.setName(label + "_AuxModel");

	filenameTrainingDataAuxModel = name + "_aux.csv";

	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);


}


void GeneralizedDerivativeEnhancedModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
	auxiliaryModel.setBoxConstraints(boxConstraintsInput);
}

void GeneralizedDerivativeEnhancedModel::setDimension(unsigned int dim){

	dimension = dim;
	data.setDimension(dim);
	auxiliaryModel.setDimension(dim);
}


void GeneralizedDerivativeEnhancedModel::setNameOfInputFile(string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;
	auxiliaryModel.setNameOfInputFile(filenameTrainingDataAuxModel);

}


void GeneralizedDerivativeEnhancedModel::setWriteWarmStartFileFlag(bool flag){
	auxiliaryModel.setWriteWarmStartFileFlag(flag);

}
void GeneralizedDerivativeEnhancedModel::setReadWarmStartFileFlag(bool flag){
	auxiliaryModel.setReadWarmStartFileFlag(flag);

}



void GeneralizedDerivativeEnhancedModel::setNameOfHyperParametersFile(string filename){
	assert(isNotEmpty(filename));
	filenameHyperparameters = filename;


}
void GeneralizedDerivativeEnhancedModel::setNumberOfTrainingIterations(unsigned int N){

	numberOfTrainingIterations = N;
	auxiliaryModel.setNumberOfTrainingIterations(N);
	output.printMessage("Number of training iterations is set to ", numberOfTrainingIterations);

}

void GeneralizedDerivativeEnhancedModel::setThetaFactor(double value){
	assert(value>=1.0);
	thetaFactor = value;
}

void GeneralizedDerivativeEnhancedModel::setDirectionalDerivativesOn(void){
	ifDirectionalDerivativesAreUsed = true;
}



void GeneralizedDerivativeEnhancedModel::readData(void){

	assert(isNotEmpty(filenameDataInput));

	if(ifDirectionalDerivativesAreUsed){
		data.setDirectionalDerivativesOn();
	}
	else{
		data.setGradientsOn();
	}

	data.readData(filenameDataInput);

	numberOfSamples = data.getNumberOfSamples();
	ifDataIsRead = true;

	prepareTrainingDataForTheKrigingModel();
	auxiliaryModel.readData();

}

void GeneralizedDerivativeEnhancedModel::prepareTrainingDataForTheKrigingModel(void){

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




void GeneralizedDerivativeEnhancedModel::normalizeData(void){
	data.normalize();
	ifNormalized = true;

	auxiliaryModel.normalizeData();

}

void GeneralizedDerivativeEnhancedModel::initializeCorrelationFunction(void){

	assert(dimension>0);

	if(!ifCorrelationFunctionIsInitialized){
		correlationFunction.setDimension(dimension);
		correlationFunction.initialize();

		differentiatedCorrelationFunction.setDimension(dimension);
		differentiatedCorrelationFunction.initialize();

		ifCorrelationFunctionIsInitialized = true;
	}


}


void GeneralizedDerivativeEnhancedModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);

	output.printMessage("Initializing the generalized gradient enhanced Kriging model...");

	resetDataObjects();

	initializeCorrelationFunction();

	numberOfHyperParameters = dimension;

	auxiliaryModel.setDimension(dimension);
	auxiliaryModel.initializeSurrogateModel();

	updateAuxilliaryFields();


	ifInitialized = true;



}
void GeneralizedDerivativeEnhancedModel::printSurrogateModel(void) const{

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
void GeneralizedDerivativeEnhancedModel::printHyperParameters(void) const{

	std::cout<<"Hyperparameters of the GRADIENT_ENHANCED model...\n";
	theta.print("theta");
	gamma.print("gamma");
	printScalar(thetaFactor);


}
void GeneralizedDerivativeEnhancedModel::saveHyperParameters(void) const{

	assert(theta.size() > 0);
	assert(gamma.size() > 0);
	assert(isNotEmpty(filenameHyperparameters));
	rowvec saveBuffer(2*dimension+1);

	for(unsigned int i=0; i<dimension; i++){
		saveBuffer(i) = theta(i);

	}
	for(unsigned int i=0; i<dimension; i++){
		saveBuffer(i+dimension) = gamma(i);
	}
	saveBuffer(2*dimension) = thetaFactor;

	saveBuffer.save(filenameHyperparameters, csv_ascii);

}
void GeneralizedDerivativeEnhancedModel::loadHyperParameters(void){


}

void GeneralizedDerivativeEnhancedModel::updateCorrelationFunctions(void){

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


void GeneralizedDerivativeEnhancedModel::setHyperParameters(vec parameters){

	assert(parameters.size() == 2*dimension);
	theta = parameters.head(dimension);
	gamma = parameters.tail(dimension);

	correlationFunction.setHyperParameters(parameters);

	differentiatedCorrelationFunction.setHyperParameters(thetaFactor*theta);


}



void GeneralizedDerivativeEnhancedModel::assembleLinearSystem(void){

	assert(ifDataIsRead);
	assert(ifNormalized);

	calculateIndicesOfSamplesWithActiveDerivatives();

	generateWeightingMatrix();
	calculatePhiMatrix();

	calculateBeta0();
	generateRhsForRBFs();

}

void GeneralizedDerivativeEnhancedModel::calculateBeta0(void) {

	output.printMessage("Number of samples = ", numberOfSamples);
	vec ys = data.getOutputVector();
	double sumys = sum(ys);
	beta0 = sumys / numberOfSamples;
	output.printMessage("beta0 = ", beta0);
}

void GeneralizedDerivativeEnhancedModel::solveLinearSystem(void) {

	assert(numberOfSamples > 0);
	assert(ydot.size() == Phi.n_rows);


	SVDSystem  linearSystemCorrelationMatrixSVD;

	linearSystemCorrelationMatrixSVD.setMatrix(Phi);
	/* SVD decomposition R = U Sigma VT */
	linearSystemCorrelationMatrixSVD.factorize();
	linearSystemCorrelationMatrixSVD.setThresholdForSingularValues(sigmaThresholdValueForSVD);
	linearSystemCorrelationMatrixSVD.setRhs(ydot);

	weights = linearSystemCorrelationMatrixSVD.solveLinearSystem();

//	vec res = ydot - Phi*weights;
//	res.print("res");

}

void GeneralizedDerivativeEnhancedModel::updateAuxilliaryFields(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	output.printMessage("Updating auxiliary model variables...");

	auxiliaryModel.updateAuxilliaryFields();

	assembleLinearSystem();

	solveLinearSystem();
}
void GeneralizedDerivativeEnhancedModel::train(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(ifInitialized);


	trainTheta();
	determineThetaCoefficientForDualBasis();


	updateAuxilliaryFields();

	ifModelTrainingIsDone = true;

}


void GeneralizedDerivativeEnhancedModel::trainTheta(void){

	assert(dimension>0);
	auxiliaryModel.train();


	vec hyperparameters = auxiliaryModel.getHyperParameters();
	assert(hyperparameters.size() == 2*dimension);

	theta = hyperparameters.head(dimension);
	gamma = hyperparameters.tail(dimension);


	//	theta.print("theta");
	//	gamma.print("gamma");


	correlationFunction.setHyperParameters(hyperparameters);


	vec thetaForDualBasis = thetaFactor*theta;
	differentiatedCorrelationFunction.setHyperParameters(thetaForDualBasis);


}

void GeneralizedDerivativeEnhancedModel::determineThetaCoefficientForDualBasis(void){

	assert(ifDataIsRead);
	assert(dimension>0);
	assert(theta.size() == dimension);
	assert(gamma.size() == dimension);
	assert(boxConstraints.areBoundsSet());

	output.printMessage("Estimation for the theta amplification factor...");

	GeneralizedDerivativeEnhancedModel auxiliaryModelForThetaCoefficient;
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
	//	hyperParameters.print("hyperparams");
	assert(hyperParameters.size() == 2*dimension);

	double exponentStart = 0.0;
	double exponentEnd   = 3.0;
	double deltaExponent = (exponentEnd-exponentStart)/ numberOfIterationsToDetermineThetaFactor;

	double bestMSE = LARGE;
	double bestFactor = 1.0;


	double exponent = exponentStart;


	for(unsigned int i=0; i<numberOfIterationsToDetermineThetaFactor; i++){


		double valueToTry = pow(10.0,exponent);
		//		printScalar(valueToTry);

		auxiliaryModelForThetaCoefficient.setThetaFactor(valueToTry);
		auxiliaryModelForThetaCoefficient.setHyperParameters(hyperParameters);

		auxiliaryModelForThetaCoefficient.resetPhiMatrix();
		auxiliaryModelForThetaCoefficient.calculatePhiMatrix();
		auxiliaryModelForThetaCoefficient.solveLinearSystem();
		auxiliaryModelForThetaCoefficient.tryOnTestData();

		double MSE = auxiliaryModelForThetaCoefficient.generalizationError;
//		printScalar(MSE);
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


double GeneralizedDerivativeEnhancedModel::interpolate(rowvec x) const{

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

		if(ifDirectionalDerivativesAreUsed){

			rowvec diffDirection = data.getRowDifferentiationDirection(index);
			r(N+i) = differentiatedCorrelationFunction.computeCorrelationDot(xi, x, diffDirection);
		}
		else{

			rowvec grad = gradientsMatrix.row(index);
			rowvec diffDirection = makeUnitVector(grad);
			r(N+i) = differentiatedCorrelationFunction.computeCorrelationDot(xi, x, diffDirection);
		}


		sum +=weights(N+i)*r(N+i);
	}


	return sum + beta0;
}



void GeneralizedDerivativeEnhancedModel::interpolateWithVariance(rowvec xp,double *fTilde,double *ssqr) const{

	auxiliaryModel.interpolateWithVariance(xp, fTilde, ssqr);
	*fTilde = interpolate(xp);


}


void GeneralizedDerivativeEnhancedModel::setTargetForSampleWeights(double value){
	targetForSampleWeights = value;
	ifTargetForSampleWeightsIsSet= true;
}


void GeneralizedDerivativeEnhancedModel::calculateIndicesOfSamplesWithActiveDerivatives(void){

	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	assert(dimension>0);

	indicesOfSamplesWithActiveDerivatives.clear();


	if(ifDirectionalDerivativesAreUsed){

		vec directionalDerivatives = data.getDirectionalDerivativesVector();
		assert(directionalDerivatives.size() == numberOfSamples);

		for(unsigned int i=0; i<numberOfSamples; i++){
			double absDirectionalDerivative = fabs(directionalDerivatives(i));

			if(absDirectionalDerivative < EPSILON){
				indicesOfSamplesWithActiveDerivatives.push_back(i);

			}

		}

		unsigned int howManyActiveSamples = indicesOfSamplesWithActiveDerivatives.size();
		output.printMessage("Number of samples with active directional derivatives = ",howManyActiveSamples);

		string msg = "Indices of samples with active directional derivatives";
		output.printList(indicesOfSamplesWithActiveDerivatives, msg);


	}else{

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


	}





	ifActiveDeritiveSampleIndicesAreCalculated = true;
}


void GeneralizedDerivativeEnhancedModel::generateWeightingMatrix(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);


	if(ifVaryingSampleWeights){
		generateSampleWeights();


		unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();

		unsigned int sizeOfWeightMatrix = numberOfSamples +  howManySamplesHaveDerivatives;

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

void GeneralizedDerivativeEnhancedModel::generateRhsForRBFs(void){

	assert(ifDataIsRead);
	assert(ifActiveDeritiveSampleIndicesAreCalculated);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int N    = numberOfSamples;

	unsigned int sizeOfRhs = N + Ndot;

	ydot = zeros<vec>(sizeOfRhs);
	vec y           = data.getOutputVector();

	/* first functional values */
	for(unsigned int i=0; i<N; i++){
		ydot(i) = y(i) - beta0;
	}

	/* then directional derivatives */
	if(ifDirectionalDerivativesAreUsed){

		vec directionalDerivatives = data.getDirectionalDerivativesVector();
		for(unsigned int i=0; i<Ndot; i++){
			ydot(i+N) = directionalDerivatives(i);
		}

	}
	else{

		mat gradients 	= data.getGradientMatrix();
		for(unsigned int i=0; i<Ndot; i++){
			unsigned int indx = indicesOfSamplesWithActiveDerivatives[i];
			rowvec grad = gradients.row(indx);
			rowvec d    = makeUnitVector(grad);

			double directionalDerivative = dot(d,grad);

			ydot(i+N) = directionalDerivative;
		}
	}

	if(ifVaryingSampleWeights){
		assert(weightMatrix.n_rows == sizeOfRhs);
		assert(weightMatrix.n_cols == sizeOfRhs);
		ydot = weightMatrix*ydot;
	}


}

void GeneralizedDerivativeEnhancedModel::calculatePhiEntriesForFunctionValues(void) {

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


	}

	if(ifDirectionalDerivativesAreUsed){

		for (unsigned int i = 0; i < N; i++) {
			rowvec xi = data.getRowX(i);

			for (unsigned int j = 0; j < Ndot;j++) {
				unsigned int index = indicesOfSamplesWithActiveDerivatives[j];
				rowvec xAtindex = data.getRowX(index);
				rowvec d = data.getRowDifferentiationDirection(index);
				Phi(i, N + j) = differentiatedCorrelationFunction.computeCorrelationDot(xAtindex, xi,d);
			}
		}

	}

	else{

		for (unsigned int i = 0; i < N; i++) {
			rowvec xi = data.getRowX(i);
			for (unsigned int j = 0; j < Ndot;j++) {
				unsigned int index = indicesOfSamplesWithActiveDerivatives[j];
				rowvec xAtIndex = data.getRowX(index);
				rowvec g = gradients.row(index);
				rowvec d = makeUnitVector(g);

				Phi(i, N + j) = differentiatedCorrelationFunction.computeCorrelationDot(xAtIndex, xi,d);
			}

		}

	}

}





void GeneralizedDerivativeEnhancedModel::calculatePhiEntriesForDerivatives(void) {

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

void GeneralizedDerivativeEnhancedModel::calculatePhiEntriesForDerivativesDirectionalDerivatives(void) {

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

		for (unsigned int j = 0; j < Ndot;j++) {
			unsigned int index = indicesOfSamplesWithActiveDerivatives[j];
			rowvec xAtindex = data.getRowX(index);
			rowvec directionBasis = data.getRowDifferentiationDirection(index);

			Phi(N + i, N + j) = differentiatedCorrelationFunction.computeCorrelationDotDot(
					xAtindex, xAtsampleIndex, directionBasis, directionAtSample);
		}

	}
}



void GeneralizedDerivativeEnhancedModel::calculatePhiMatrix(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);


	unsigned int howManySamplesHaveDerivatives = indicesOfSamplesWithActiveDerivatives.size();
	unsigned int howManyTotalDataPoints = numberOfSamples + howManySamplesHaveDerivatives;
	unsigned int howManyBasisFunctions = numberOfSamples + howManySamplesHaveDerivatives;

	Phi = zeros<mat>(howManyTotalDataPoints, howManyBasisFunctions);

	calculatePhiEntriesForFunctionValues();

	if(ifDirectionalDerivativesAreUsed){

		calculatePhiEntriesForDerivativesDirectionalDerivatives();
	}
	else{

		calculatePhiEntriesForDerivatives();
	}




	if(ifVaryingSampleWeights){

		unsigned int sizeOfWeightMatrix = howManyTotalDataPoints;
		assert(weightMatrix.n_rows == sizeOfWeightMatrix);
		Phi = weightMatrix*Phi;
	}

}

void GeneralizedDerivativeEnhancedModel::resetPhiMatrix(void){
	Phi.reset();

}

bool GeneralizedDerivativeEnhancedModel::checkResidual(void) const{

	vec residual = ydot - Phi*weights;
	if(norm(residual) < 10E-3) return true;
	else return false;
}


bool GeneralizedDerivativeEnhancedModel::checkPhiMatrix(void){


	unsigned int N    = numberOfSamples;
	unsigned int Ndot = indicesOfSamplesWithActiveDerivatives.size();

	if(weights.size() == 0){
		weights = ones<vec>(N+Ndot);
	}



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

	if(ifDirectionalDerivativesAreUsed){

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
			for(unsigned int j=0; j<N+Ndot;j++){
				sum +=Phi(N+i,j)*weights(j);
			}

			double error = fabs(sum - fdValue);
			if(error > 10E-05) return false;

		}

	}

	else{

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


	}






	return true;
}

void GeneralizedDerivativeEnhancedModel::resetDataObjects(void){

	Phi.reset();
	weights.reset();
	weightMatrix.reset();
	indicesOfSamplesWithActiveDerivatives.clear();
	beta0 = 0.0;
	auxiliaryModel.resetDataObjects();


}

void GeneralizedDerivativeEnhancedModel::addNewSampleToData(rowvec newsample){

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


void GeneralizedDerivativeEnhancedModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(false);

}


void GeneralizedDerivativeEnhancedModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();
	updateAuxilliaryFields();

}


unsigned int GeneralizedDerivativeEnhancedModel::getNumberOfSamplesWithActiveGradients(void) const{
	return indicesOfSamplesWithActiveDerivatives.size();

}


mat GeneralizedDerivativeEnhancedModel::getPhiMatrix(void) const{
	return Phi;
}
mat GeneralizedDerivativeEnhancedModel::getWeightMatrix(void) const{
	return weightMatrix;
}

vec GeneralizedDerivativeEnhancedModel::getSampleWeightsVector(void) const{
	return sampleWeights;
}
mat GeneralizedDerivativeEnhancedModel::getGradient(void) const{
	assert(ifDataIsRead);
	return data.getGradientMatrix();
}
