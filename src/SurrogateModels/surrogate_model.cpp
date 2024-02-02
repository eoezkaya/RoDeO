/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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
 * General Public License along with RoDEO.
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
#include <cassert>

#include "./INCLUDE/surrogate_model.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"
#include "../INCLUDE/Rodeo_globals.hpp"

#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;



SurrogateModel::SurrogateModel(){}

void SurrogateModel::setName(std::string nameInput){

	assert(isNotEmpty(nameInput));

	name = nameInput;

	filenameHyperparameters = name + "_hyperparameters.csv";

}

string SurrogateModel::getName(void) const{
	return name;
}


void SurrogateModel::setDimension(unsigned int dim){

	dimension = dim;
}


void SurrogateModel::setNumberOfThreads(unsigned int n){

	numberOfThreads = n;

}


void SurrogateModel::setGradientsOn(void){

	data.setGradientsOn();
	ifHasGradientData = true;


}

void SurrogateModel::setGradientsOff(void){

	data.setGradientsOff();
	ifHasGradientData = false;

}

bool SurrogateModel::areGradientsOn(void) const{

	return ifHasGradientData;

}

void SurrogateModel::setWriteWarmStartFileFlag(bool flag){

	ifWriteWarmStartFile = flag;
}

void SurrogateModel::setReadWarmStartFileFlag(bool flag){
	ifReadWarmStartFile = flag;

}

void SurrogateModel::setDisplayOn(void){

	data.setDisplayOn();
	output.ifScreenDisplay = true;

}

void SurrogateModel::setDisplayOff(void){

	data.setDisplayOff();
	output.ifScreenDisplay = false;

}



std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return filenameHyperparameters;

}


std::string SurrogateModel::getNameOfInputFile(void) const{

	return filenameDataInput;

}


unsigned int SurrogateModel::getDimension(void) const{
	return dimension;
}
unsigned int SurrogateModel::getNumberOfSamples(void) const{
	return numberOfSamples;
}

mat SurrogateModel::getRawData(void) const{
	return data.getRawData();
}

mat SurrogateModel::getX(void) const{

	return data.getInputMatrix();
}

vec SurrogateModel::gety(void) const{

	return data.getOutputVector();

}


void SurrogateModel::printData(void) const{
	data.print();
}

void SurrogateModel::readDataTest(void){

	assert(isNotEmpty(filenameDataInputTest));

	data.readDataTest(filenameDataInputTest);

	ifTestDataIsRead = true;
}

unsigned int SurrogateModel::countHowManySamplesAreWithinBounds(vec lb, vec ub){

	assert(ifDataIsRead);

	mat trainingData;
	trainingData = data.getRawData();

	unsigned int N   = data.getNumberOfSamples();
	unsigned int dim = data.getDimension();

	unsigned int counter = 0;
	for(unsigned int i=0; i<N; i++){

		rowvec sample     = trainingData.row(i);
		rowvec dv         = sample.head(dim);
		vec x = trans(dv);

		if(isBetween(x,lb,ub)){
			counter++;
		}

	}

	return counter;
}


void SurrogateModel::reduceTrainingData(vec lb, vec ub) const{

	assert(ifDataIsRead);

	mat trainingData;
	trainingData = data.getRawData();

#if 0
	trainingData.print();
#endif
	unsigned int dim = data.getDimension();
	unsigned int N   = data.getNumberOfSamples();

	vector<int> samplesThatCanBeRemoved;

	for(unsigned int i=0; i<N; i++){

		rowvec sample     = trainingData.row(i);
		rowvec dv         = sample.head(dim);
		vec x = trans(dv);

		if(!isBetween(x,lb,ub)){
			samplesThatCanBeRemoved.push_back(i);
		}

	}

	unsigned int howManySamplesToBeRemoved = samplesThatCanBeRemoved.size();

	if(howManySamplesToBeRemoved>0){


		uvec indicesToRemove(howManySamplesToBeRemoved);

		for(unsigned int i=0; i<howManySamplesToBeRemoved; i++){
			indicesToRemove(i) = samplesThatCanBeRemoved.at(i);
		}


		trainingData.shed_rows(indicesToRemove);
		trainingData.save(filenameDataInput, csv_ascii);

	}


}



//void SurrogateModel::reduceTrainingData(unsigned howManySamples, double targetValue) const{
//
//	assert(ifDataIsRead);
//	Bounds boxConstraints = data.getBoxConstraints();
//	assert(boxConstraints.areBoundsSet());
//
//	unsigned int dim = data.getDimension();
//	unsigned int N   = data.getNumberOfSamples();
//
//	mat trainingData;
//	trainingData = data.getRawData();
//
//	vec lb = boxConstraints.getLowerBounds();
//	vec ub = boxConstraints.getUpperBounds();
//
//	vector<pair<unsigned int, double >> samplesThatCanBeRemoved;
//
//	for(unsigned int i=0; i<N; i++){
//
//		rowvec sample     = trainingData.row(i);
//		rowvec dv         = sample.head(dim);
//		double value      = sample(dim);
//		vec x = trans(dv);
//
//		if(!isBetween(x,lb,ub)){
//
//			pair<unsigned int, double> sampleToRemove;
//
//			sampleToRemove.first = i;
//			sampleToRemove.second = fabs(value - targetValue);
//
//			samplesThatCanBeRemoved.push_back(sampleToRemove);
//		}
//	}
//
//	unsigned int Nreduced = samplesThatCanBeRemoved.size();
//
//	if(Nreduced == 0){
//		howManySamples = 0;
//
//	}
//
//	for(unsigned int i=0; i<Nreduced; i++){
//
//		for(unsigned int j=i; j<Nreduced; j++){
//
//			pair<unsigned int, double> sample1;
//			pair<unsigned int, double> sample2;
//
//			sample1 = samplesThatCanBeRemoved.at(i);
//			sample2 = samplesThatCanBeRemoved.at(j);
//
//			double value1 = sample1.second;
//			double value2 = sample2.second;
//
//			if(value2 > value1){
//
//				pair<unsigned int, double> temp;
//				temp = samplesThatCanBeRemoved.at(i);
//				samplesThatCanBeRemoved.at(i) = samplesThatCanBeRemoved.at(j);
//				samplesThatCanBeRemoved.at(j) = temp;
//			}
//		}
//	}
//
//	uvec indicesToRemove(howManySamples);
//
//	for(unsigned int i=0; i<howManySamples; i++){
//		indicesToRemove(i) = samplesThatCanBeRemoved.at(i).first;
//	}
//
//	unsigned int Nleft = N - howManySamples;
//	mat writeBuffer(Nleft,trainingData.n_cols);
//
//	trainingData.shed_rows(indicesToRemove);
//	trainingData.save(filenameDataInput, csv_ascii);
//
//}


void SurrogateModel::normalizeDataTest(void){

	data.normalizeSampleInputMatrixTest();
	ifNormalizedTestData = true;

}

void SurrogateModel::updateAuxilliaryFields(void){

}


vec SurrogateModel::interpolateVector(mat X) const{

	assert(X.max() <= 1.0/data.getDimension());
	assert(X.min() >= 0.0);

	unsigned int N = X.n_rows;
	vec results(N);

	for(unsigned int i=0; i<N; i++){

		rowvec xp = X.row(i);
		results(i) = interpolate(xp);

	}

	return results;
}


double SurrogateModel::calculateInSampleError(void) const{

	assert(ifInitialized);


	vec fTildeValues = interpolateVector(data.getInputMatrix());
	vec fExact = data.getOutputVector();
	vec diff = fTildeValues - fExact;

	double L2normDiff = norm(diff,2);
	double squaredError = L2normDiff*L2normDiff;

	return squaredError/data.getNumberOfSamples();

}



double SurrogateModel::calculateOutSampleError(void){

	assert(ifHasTestData);
	assert(data.ifTestDataHasFunctionValues);

	tryOnTestData();

	vec squaredErrors = testResults.col(data.getDimension()+2);

	return mean(squaredErrors);

}

void SurrogateModel::saveTestResults(void) const{

	assert(isNotEmpty(filenameTestResults));
	field<std::string> header(testResults.n_cols);

	unsigned int dim = data.getDimension();

	for(unsigned int i=0; i<dim; i++){
		header(i) ="x"+std::to_string(i+1);
	}

	header(dim)   = "Estimated value";

	if(data.ifTestDataHasFunctionValues){

		header(dim+1) = "True value";
		header(dim+2) = "Squared Error";
	}



	testResults.save( csv_name(filenameTestResults, header) );

	output.printMessage("Writing results to the file = ", filenameTestResults);

}

void SurrogateModel::printSurrogateModel(void) const{
	data.print();
}
rowvec SurrogateModel::getRowX(unsigned int index) const{
	return data.getRowX(index);
}

rowvec SurrogateModel::getRowXRaw(unsigned int index) const{
	return data.getRowXRaw(index);
}

void SurrogateModel::setNameOfInputFileTest(string filename){

	assert(isNotEmpty(filename));
	filenameDataInputTest = filename;

	ifHasTestData = true;
}

void SurrogateModel::setNameOfOutputFileTest(string filename){

	assert(isNotEmpty(filename));
	filenameTestResults = filename;
}

void SurrogateModel::tryOnTestData(void){

	assert(ifNormalizedTestData);


	output.printMessage("Trying surrogate model on test data...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfEntries;

	if(data.ifTestDataHasFunctionValues){

		numberOfEntries = dim + 3;
	}
	else{

		numberOfEntries = dim + 1;
	}

	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();
	vec squaredError(numberOfTestSamples, fill::zeros);


	mat results(numberOfTestSamples,numberOfEntries);

	vec fExact;
	if(data.ifTestDataHasFunctionValues){
		fExact = data.getOutputVectorTest();
	}

	mat XTest = data.getInputMatrixTest();


	for(unsigned int i=0; i<numberOfTestSamples; i++){

		rowvec xp          = data.getRowXTest(i);
		rowvec dataRow     = data.getRowXRawTest(i);

		output.printMessage("\n");
		rowvec x = dataRow.head(dimension);
		output.printMessage("x = ",x);


		double fTilde = interpolate(xp);


		rowvec sample = x;
		addOneElement<rowvec>(sample,fTilde);

		if(data.ifTestDataHasFunctionValues){

			addOneElement<rowvec>(sample,fExact(i));
			double error = pow((fExact(i) - fTilde),2.0);
			squaredError(i) = error;
			addOneElement<rowvec>(sample,error);

			output.printMessage("f(x) = ",fExact(i), "estimate = ", fTilde);
			output.printMessage("Squared error = ", error);
			output.printMessage("\n");
		}
		else{

			output.printMessage("fTilde = ",fTilde);
		}

		results.row(i) = sample;
	}

	if(data.ifTestDataHasFunctionValues){

		generalizationError = mean(squaredError);
		standardDeviationOfGeneralizationError = stddev(squaredError);
	}


	testResults = results;
}

void SurrogateModel::printGeneralizationError(void) const{

	if(generalizationError>0.0){

		unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();
		string msg = "Generalization error (MSE) = " + convertToString(generalizationError,15) + " ";
		msg += "RMSE = " + convertToString(sqrt(generalizationError),15) + " ";
		msg += "(Evaluated at " + std::to_string(numberOfTestSamples) + " samples)";
		output.printMessage(msg);
		msg = "standard deviation of the MSE = " + std::to_string(standardDeviationOfGeneralizationError);
		output.printMessage(msg);

	}


}

/* This function generates weights for each sample according to a target value */
void SurrogateModel::generateSampleWeights(void){

	output.printMessage("Generating sample weights...");

	assert(ifDataIsRead);
	assert(numberOfSamples>0);

	sampleWeights = zeros<vec>(numberOfSamples);

	vec y = data.getOutputVector();
	double targetValue = min(y);

	if(ifTargetForSampleWeightsIsSet){
		targetValue = targetForSampleWeights;

		output.printMessage("Target value for the sample weights = ", targetValue);

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

void SurrogateModel::printSampleWeights(void) const{

	assert(numberOfSamples>0);
	assert(sampleWeights.size() == numberOfSamples);

	vec y = data.getOutputVector();
	for(unsigned int i=0; i<numberOfSamples; i++){
		std::cout<<"y("<<i<<") = "<<y(i)<<", w = " << sampleWeights(i) << "\n";
	}
}

void SurrogateModel::removeVeryCloseSamples(const Design& globalOptimalDesign){

	assert(ifDataIsRead);
	data.removeVeryCloseSamples(globalOptimalDesign);

}


