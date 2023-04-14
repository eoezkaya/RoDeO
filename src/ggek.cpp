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


mat GGEKModel::getGradient(void) const{
	assert(ifDataIsRead);
	return data.getGradientMatrix();
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
void GGEKModel::updateAuxilliaryFields(void){



}
void GGEKModel::train(void){



}
double GGEKModel::interpolate(rowvec x) const{



}
void GGEKModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{




}

void GGEKModel::addNewSampleToData(rowvec newsample){


}
void GGEKModel::addNewLowFidelitySampleToData(rowvec newsample){



}

void GGEKModel::setTargetForSampleWeights(double value){
	targetForSampleWeights = value;
	ifTargerForSampleWeightsIsSet= true;
}

mat GGEKModel::getWeightMatrix(void) const{
	return weightMatrix;
}

vec GGEKModel::getSampleWeightsVector(void) const{
	return sampleWeights;
}


/* This function generates weights for each sample according to a target value */
void GGEKModel::generateSampleWeights(void){

	sampleWeights = zeros<vec>(numberOfSamples);
	vec y = data.getOutputVector();
	double targetValue = min(y);

	if(ifTargerForSampleWeightsIsSet){
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
		gradVec.print();

		if(!gradVec.is_zero()){

			indicesOfSamplesWithActiveDerivatives.push_back(i);
		}


	}

//	for (std::vector<int>::const_iterator i = indicesOfSamplesWithActiveDerivatives.begin(); i != indicesOfSamplesWithActiveDerivatives.end(); ++i)
//	    std::cout << *i << ' ';

}


//void GGEKModel::generateWeightingMatrix(void){
//
//	generateSampleWeights();
//
//	weightMatrix = zeros<mat>(2*N, 2*N);
//
//	for(unsigned int i=0; i<N; i++){
//		weightMatrix(i,i) = sampleWeights(i);
//	}
//	for(unsigned int i=N; i<2*N; i++){
//		weightMatrix(i,i) = sampleWeights(i-N)*0.5;
//	}
//
//}



