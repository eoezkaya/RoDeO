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

#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


KrigingModel::KrigingModel():SurrogateModel(){}

void KrigingModel::setDimension(unsigned int dim){

	dimension = dim;
	linearModel.setDimension(dim);
}

void KrigingModel::setNameOfInputFile(std::string filename){

	assert(isNotEmpty(filename));

	filenameDataInput = filename;
	linearModel.setNameOfInputFile(filename);

}

void KrigingModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void KrigingModel::setNameOfHyperParametersFile(std::string filename){

	assert(isNotEmpty(filename));
	hyperparameters_filename = filename;

}

void KrigingModel::setDisplayOn(void){
	data.setDisplayOn();
	linearModel.setDisplayOn();
	output.ifScreenDisplay = true;
}

void KrigingModel::setDisplayOff(void){
	data.setDisplayOff();
	linearModel.setDisplayOff();
	output.ifScreenDisplay = false;
}

void KrigingModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	if(ifUsesLinearRegression) linearModel.setBoxConstraints(boxConstraintsInput);
	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
}

void KrigingModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.readData(filenameDataInput);

	if(ifUsesLinearRegression) linearModel.readData();
	numberOfSamples = data.getNumberOfSamples();
	ifDataIsRead = true;

}

void KrigingModel::normalizeData(void){
	assert(ifDataIsRead);
	data.normalize();
	if(ifUsesLinearRegression) linearModel.normalizeData();
	ifNormalized = true;
}

void KrigingModel::initializeSurrogateModel(void){

	assert(dimension>0);
	assert(numberOfSamples>0);
	assert(ifDataIsRead);
	assert(ifNormalized);


	output.printMessage("Initializing the Kriging model...");

	mat X = data.getInputMatrix();

	correlationFunction.setInputSampleMatrix(X);

	if(!ifCorrelationFunctionIsInitialized){
		correlationFunction.initialize();
		ifCorrelationFunctionIsInitialized = true;
	}


	numberOfHyperParameters = 2*dimension;


	R_inv_ys_min_beta = zeros<vec>(numberOfSamples);
	R_inv_I= zeros<vec>(numberOfSamples);
	R_inv_ys = zeros<vec>(numberOfSamples);
	vectorOfOnes= ones<vec>(numberOfSamples);

	if(ifUsesLinearRegression){

		output.printMessage("Linear model is active for the Kriging model...");

		linearModel.initializeSurrogateModel();
		vec ys = data.getOutputVector();
		mat X = data.getInputMatrix();
		vec ysLinearRegression = linearModel.interpolateAll(X);
		ys = ys - ysLinearRegression;
		data.setOutputVector(ys);

	}

	updateAuxilliaryFields();

	ifInitialized = true;

}

void KrigingModel::updateAuxilliaryFields(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(ifCorrelationFunctionIsInitialized);
	assert(numberOfSamples>0);
	assert(dimension>0);

	correlationFunction.computeCorrelationMatrix();
	mat R = correlationFunction.getCorrelationMatrix();

	linearSystemCorrelationMatrix.setMatrix(R);

	/* Cholesky decomposition R = L L^T */

	linearSystemCorrelationMatrix.factorize();


	R_inv_ys = zeros<vec>(numberOfSamples);
	R_inv_I = zeros<vec>(numberOfSamples);
	R_inv_ys_min_beta = zeros<vec>(numberOfSamples);
	beta0 = 0.0;
	sigmaSquared = 0.0;

	if(linearSystemCorrelationMatrix.isFactorizationDone()){

		/* solve R x = ys */
		vec ys = data.getOutputVector();

		R_inv_ys = linearSystemCorrelationMatrix.solveLinearSystem(ys);

		/* solve R x = I */

		R_inv_I = linearSystemCorrelationMatrix.solveLinearSystem(vectorOfOnes);

		beta0 = (1.0/dot(vectorOfOnes,R_inv_I)) * (dot(vectorOfOnes,R_inv_ys));

		vec ys_min_betaI = ys - beta0*vectorOfOnes;

		/* solve R x = ys-beta0*I */

		R_inv_ys_min_beta = linearSystemCorrelationMatrix.solveLinearSystem( ys_min_betaI);

		sigmaSquared = (1.0 / numberOfSamples) * dot(ys_min_betaI, R_inv_ys_min_beta);

	}

#if 0

	checkAuxilliaryFields();

#endif

}


void KrigingModel::printHyperParameters(void) const{

	unsigned int dim = data.getDimension();

	std::cout<<"Hyperparameters of the Kriging model = \n";


	vec hyperParameters = correlationFunction.getHyperParameters();
	vec theta = hyperParameters.head(dim);
	vec gamma = hyperParameters.tail(dim);

	printVector(theta,"theta");
	printVector(gamma,"gamma");

	if(ifUsesLinearRegression){

		vec w = linearModel.getWeights();
		std::cout<<"Weights of the linear regression model = \n";
		printVector(w,"weights");

	}

}

void KrigingModel::setHyperParameters(vec parameters){


	assert(parameters.size() != 0);
	correlationFunction.setHyperParameters(parameters);


}

vec KrigingModel::getHyperParameters(void) const{

	return correlationFunction.getHyperParameters();


}



void KrigingModel::saveHyperParameters(void) const{


	if(!hyperparameters_filename.empty()){

		output.printMessage("Saving hyperparameters into the file: ", hyperparameters_filename);

		unsigned int dim = data.getDimension();

		vec saveBuffer(numberOfHyperParameters);

		saveBuffer = correlationFunction.getHyperParameters();
		saveBuffer.save(hyperparameters_filename,csv_ascii);

	}

}

void KrigingModel::loadHyperParameters(void){

	vec loadBuffer;
	bool ifLoadIsOK =  loadBuffer.load(hyperparameters_filename,csv_ascii);

	unsigned int numberOfEntriesInTheBuffer = loadBuffer.size();

	if(ifLoadIsOK && numberOfEntriesInTheBuffer == numberOfHyperParameters) {

		unsigned int dim = data.getDimension();

		vec theta = loadBuffer.head(dim);
		vec gamma = loadBuffer.tail(dim);

		correlationFunction.setTheta(theta);
		correlationFunction.setGamma(gamma);

	}

}


vec KrigingModel::getRegressionWeights(void) const{
	return linearModel.getWeights();
}
void KrigingModel::setRegressionWeights(vec weights){
	linearModel.setWeights(weights);
}

void KrigingModel::setEpsilon(double value){

	assert(value>=0);
	correlationFunction.setEpsilon(value);

}

void KrigingModel::setLinearRegressionOn(void){

	ifUsesLinearRegression  = true;

}
void KrigingModel::setLinearRegressionOff(void){

	ifUsesLinearRegression  = false;

}




/** Adds the rowvector newsample to the data of the Kriging model and updates model parameters
 * @param[in] newsample
 *
 */

void KrigingModel::addNewSampleToData(rowvec newsample){

	unsigned int dim = data.getDimension();
	assert(newsample.size() == dim+1);

	/* avoid points that are too close to each other */

	mat rawData = data.getRawData();


	bool flagTooClose= checkifTooCLose(newsample, rawData);


	if(!flagTooClose){

		appendRowVectorToCSVData(newsample, filenameDataInput);

		updateModelWithNewData();

	}
	else{

		std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

	}

}


void KrigingModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(false);

}


void KrigingModel::printSurrogateModel(void) const{
	cout << "\nKriging Surrogate model:\n";
	cout<< "Number of samples: "<<data.getNumberOfSamples()<<endl;
	cout<< "Number of input parameters: "<<data.getDimension() <<"\n";


	printf("hyperparameters_filename: %s\n",hyperparameters_filename.c_str());
	printf("input_filename: %s\n",filenameDataInput.c_str());
	printf("max_number_of_kriging_iterations = %d\n",this->numberOfTrainingIterations);

	data.print();

	correlationFunction.print();

	printHyperParameters();

#if 0
	printVector(R_inv_ys_min_beta,"R_inv_ys_min_beta");
#endif
	std::cout<<"beta0 = "<<beta0<<"\n";
#if 0
	printMatrix(correlationMatrix,"correlationMatrix");
#endif
	printf("\n");

}

void KrigingModel::resetDataObjects(void){

	R_inv_I.reset();
	R_inv_ys_min_beta.reset();
	vectorOfOnes.reset();
	beta0 = 0.0;
	sigmaSquared = 0.0;

}

void KrigingModel::resizeDataObjects(void){

	unsigned int numberOfSamples = data.getNumberOfSamples();


	R_inv_ys_min_beta.set_size(numberOfSamples);
	R_inv_ys_min_beta.fill(0.0);
	R_inv_I.set_size(numberOfSamples);
	R_inv_I.fill(0.0);
	vectorOfOnes.set_size(numberOfSamples);
	vectorOfOnes.fill(1.0);


}

void KrigingModel::checkAuxilliaryFields(void) const{

	mat R = correlationFunction.getCorrelationMatrix();
	vec ys = data.getOutputVector();
	vec ys_min_betaI = ys - beta0*vectorOfOnes;
	vec residual2 = ys_min_betaI - R*R_inv_ys_min_beta;
	printVector(residual2,"residual (ys-betaI - R * R^-1 (ys-beta0I) )");
	vec residual3 = vectorOfOnes - R*R_inv_I;
	printVector(residual3,"residual (I - R * R^-1 I)");
}

void KrigingModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();

}

double KrigingModel::interpolate(rowvec xp ) const{


	double estimateLinearRegression = 0.0;
	double estimateKriging = 0.0;

	if(ifUsesLinearRegression ){

		estimateLinearRegression = linearModel.interpolate(xp);
	}

	vec r = correlationFunction.computeCorrelationVector(xp);

	estimateKriging = beta0 + dot(r,R_inv_ys_min_beta);

	return estimateLinearRegression + estimateKriging;

}


void KrigingModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const{

	assert(ifInitialized);
	unsigned int N = data.getNumberOfSamples();
	*ftildeOutput =  interpolate(xp);

	vec R_inv_r(N);

	vec r = correlationFunction.computeCorrelationVector(xp);

	/* solve the linear system R x = r by Cholesky matrices U and L*/

	R_inv_r = linearSystemCorrelationMatrix.solveLinearSystem(r);

	double dotRTRinvR = dot(r,R_inv_r);
	double dotRTRinvI = dot(r,R_inv_I);
	double dotITRinvI = dot(vectorOfOnes,R_inv_I);

	double term1 = pow( (dotRTRinvI - 1.0 ),2.0)/dotITRinvI;

	if(fabs(dotRTRinvI - 1.0) < 10E-10) {

		term1 = 0.0;
	}

	double term2 = 1.0 - dotRTRinvR;

	if(fabs(dotRTRinvR - 1.0) < 10E-10) {

		term2 = 0.0;
	}

	double term3 = term2 + term1;

	*sSqrOutput = sigmaSquared* term3;

}

mat KrigingModel::getCorrelationMatrix(void) const{

	return linearSystemCorrelationMatrix.getMatrix();
}

double KrigingModel::calculateLikelihoodFunction(vec hyperParameters){

	assert(ifDataIsRead);
	assert(ifNormalized);

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

	correlationFunction.setHyperParameters(hyperParameters);

	updateAuxilliaryFields();

	if(linearSystemCorrelationMatrix.isFactorizationDone() == false){

		return -LARGE;
	}

	double logdetR = linearSystemCorrelationMatrix.calculateLogDeterminant();
	double NoverTwo = double(N)/2.0;

	double likelihoodValue = 0.0;

	if(sigmaSquared > 0 ){
		double logSigmaSqr = log(sigmaSquared);
		likelihoodValue = (- NoverTwo) * logSigmaSqr;
		likelihoodValue -= 0.5 * logdetR;
	}
	else{

		likelihoodValue = -LARGE;

	}

	return likelihoodValue;


}

void KrigingModel::train(void){

	assert(ifInitialized);

	unsigned int dim = data.getDimension();
	assert(dim>0);


	Bounds boxConstraintsForTheTraining(2*dim);
	vec lb(2*dim,fill::zeros);
	vec ub(2*dim);

	for(unsigned int i=0; i<dim; i++)     ub(i) = 10.0;
	for(unsigned int i=dim; i<2*dim; i++) ub(i) = 2.0;

	boxConstraintsForTheTraining.setBounds(lb,ub);
	double globalBestL1error = LARGE;

	KrigingHyperParameterOptimizer bestOptimizer;
	omp_set_num_threads(numberOfThreads);

	numberOfTrainingIterations = numberOfTrainingIterations/numberOfThreads;

#pragma omp parallel for
	for(unsigned int thread = 0; thread< numberOfThreads; thread++){

		KrigingHyperParameterOptimizer parameterOptimizer;

		parameterOptimizer.setDimension(2*dim);
		parameterOptimizer.initializeKrigingModelObject(*this);
		//		parameterOptimizer.setDisplayOn();

		parameterOptimizer.setBounds(boxConstraintsForTheTraining);
		parameterOptimizer.setNumberOfNewIndividualsInAGeneration(200*dim);
		parameterOptimizer.setNumberOfDeathsInAGeneration(180*dim);
		parameterOptimizer.setInitialPopulationSize(2*dim*100);
		parameterOptimizer.setMutationProbability(0.1);
		parameterOptimizer.setMaximumNumberOfGeneratedIndividuals(numberOfTrainingIterations);

		unsigned int numberOfGenerations = numberOfTrainingIterations/(200.0*dim);

		if(numberOfGenerations == 0){

			numberOfGenerations = 1;
		}
		parameterOptimizer.setNumberOfGenerations(numberOfGenerations);


		if(ifReadWarmStartFile){

			assert(isNotEmpty(filenameForWarmStartModelTraining));

			parameterOptimizer.setFilenameWarmStart(filenameForWarmStartModelTraining);
			parameterOptimizer.setWarmStartOn();
		}


		parameterOptimizer.optimize();

		EAIndividual bestSolution = parameterOptimizer.getSolution();
		vec optimizedHyperParameters = bestSolution.getGenes();


#pragma omp critical
		{

			double bestSolutionLikelihood = bestSolution.getObjectiveFunctionValue();
			if( bestSolutionLikelihood  < globalBestL1error){

				bestOptimizer = parameterOptimizer;
				globalBestL1error = bestSolutionLikelihood ;
			}
		}
	}

	omp_set_num_threads(1);

	if(ifWriteWarmStartFile){

		bestOptimizer.setFilenameWarmStart(filenameForWriteWarmStart);
		bestOptimizer.writeWarmRestartFile();
	}

	correlationFunction.setHyperParameters(bestOptimizer.getBestDesignVector());

	updateAuxilliaryFields();
	ifModelTrainingIsDone = true;

}

void KrigingHyperParameterOptimizer::initializeKrigingModelObject(KrigingModel input){

	assert(input.ifDataIsRead);
	assert(input.ifNormalized);
	assert(input.ifInitialized);

	KrigingModelForCalculations = input;

	ifModelObjectIsSet = true;

}

double KrigingHyperParameterOptimizer::calculateObjectiveFunctionInternal(vec& input){

	return -1.0* KrigingModelForCalculations.calculateLikelihoodFunction(input);

}
