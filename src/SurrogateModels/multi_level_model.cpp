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

#include <armadillo>
#include<cassert>

#include "LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "multi_level_method.hpp"
#include "kriging_training.hpp"
#include "vector_operations.hpp"
#include "Rodeo_globals.hpp"
#include "metric.hpp"
#include "auxiliary_functions.hpp"
using namespace arma;
using namespace std;

void MultiLevelModel::setName(std::string nameGiven){

	assert(isNotEmpty(nameGiven));
	assert(ifSurrogateModelsAreSet);

	name = nameGiven;
	string nameLowFiModel = name+"_LowFi";
	lowFidelityModel->setName(nameLowFiModel);
	string nameErrorModel = name+"_Error";
	errorModel->setName(nameErrorModel);


}

void MultiLevelModel::setDimension(unsigned int dim){

	dimension = dim;

	if(ifSurrogateModelsAreSet){

		lowFidelityModel->setDimension(dim);
		errorModel->setDimension(dim);
	}

}



void MultiLevelModel::setNameOfInputFile(string filename){}

void MultiLevelModel::setNameOfInputFileError(void){
	assert(isNotEmpty(name));
	assert(ifSurrogateModelsAreSet);

	inputFileNameError = name +"_Error.csv";
	errorModel->setNameOfInputFile(inputFileNameError);
	output.printMessage("Error data is set as ", inputFileNameError);
}


void MultiLevelModel::setNameOfHyperParametersFile(string label){

	assert(isNotEmpty(label));
	filenameHyperparameters= label;


}
void MultiLevelModel::setNumberOfTrainingIterations(unsigned int nIter){

	lowFidelityModel->setNumberOfTrainingIterations(nIter);
	errorModel->setNumberOfTrainingIterations(nIter);

	output.printMessage("Number of training iterations is set as" + numberOfTrainingIterations);

}

void MultiLevelModel::setinputFileNameHighFidelityData(string filename){
	assert(isNotEmpty(filename));
	filenameDataInput = filename;
	output.printMessage("Name for the high fidelity input data is set as " + filenameDataInput);
}

void MultiLevelModel::setinputFileNameLowFidelityData(string filename){
	assert(isNotEmpty(filename));
	filenameDataInputLowFidelity = filename;
	output.printMessage("Name for the low fidelity input data is set as " + filenameDataInputLowFidelity);
}


void MultiLevelModel::setDisplayOn(void){
	lowFidelityModel->setDisplayOn();
	output.ifScreenDisplay = true;
}

void MultiLevelModel::setDisplayOff(void){
	output.ifScreenDisplay = false;
}


void MultiLevelModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(ifSurrogateModelsAreSet);
	assert(boxConstraintsInput.areBoundsSet());

	data.setBoxConstraints(boxConstraintsInput);
	dataLowFidelity.setBoxConstraints(boxConstraintsInput);
	lowFidelityModel->setBoxConstraints(boxConstraintsInput);
	errorModel->setBoxConstraints(boxConstraintsInput);
	boxConstraints = boxConstraintsInput;
	ifBoxConstraintsAreSet = true;
}


void MultiLevelModel::readData(void){

	assert(dimension>0);
	assert(ifSurrogateModelsAreSet);
	assert(isNotEmpty(filenameDataInput));
	assert(isNotEmpty(filenameDataInputLowFidelity));

	output.printMessage("Reading data for the multilevel model...");

	data.readData(filenameDataInput);
	numberOfSamples = data.getNumberOfSamples();

	dataLowFidelity.readData(filenameDataInputLowFidelity);
	numberOfSamplesLowFidelity = dataLowFidelity.getNumberOfSamples();
	lowFidelityModel->readData();

	rawDataLowFidelity = dataLowFidelity.getRawData();
	rawDataHighFidelity = data.getRawData();

	ifDataIsRead = true;
	prepareAndReadErrorData();
	output.printMessage("Reading data for the multilevel model is done...");

}


void MultiLevelModel::prepareAndReadErrorData(void){

	assert(dimension>0);
	assert(ifDataIsRead);
	assert(ifSurrogateModelsAreSet);
	assert(numberOfSamples>0);
	assert(numberOfSamplesLowFidelity>0);
	assert(numberOfSamplesLowFidelity >= numberOfSamples);

	output.printMessage("Preparing error data...");

	determineAlpha();

	setNameOfInputFileError();

	rawDataError.reset();

	if(modelIDError == ORDINARY_KRIGING){
		rawDataError = zeros<mat>(numberOfSamples,dimension+1);
	}
	if(modelIDError == TANGENT){
		rawDataError = zeros<mat>(numberOfSamples,2*dimension+2);
	}

	for(unsigned int i = 0; i < numberOfSamples; i++){

		unsigned int indexLowFi = findIndexHiFiToLowFiData(i);


		double error = rawDataHighFidelity(i,dimension) - alpha*rawDataLowFidelity(indexLowFi,dimension);

		for(unsigned int j = 0; j < dimension; j++){
			rawDataError(i,j) = rawDataHighFidelity(i,j);
		}
		rawDataError(i,dimension) = error;

	}

	if(modelIDError == TANGENT){

		for(unsigned int i = 0; i < numberOfSamples; i++){

			unsigned int indexLowFi = findIndexHiFiToLowFiData(i);

			double errorDirectionalDerivative = rawDataHighFidelity(i,dimension+1) - alpha*rawDataLowFidelity(indexLowFi,dimension+1);

			rawDataError(i,dimension+1) = errorDirectionalDerivative;

			for(unsigned int j = 0; j < dimension; j++){
				rawDataError(i,dimension+2+j) = rawDataHighFidelity(i,dimension+2+j);
			}
		}

	}


	rawDataError.save(inputFileNameError, csv_ascii);

	errorModel->readData();

	output.printMessage("Error data is saved in " + inputFileNameError);

	ifErrorDataIsSet = true;

	output.printMessage("Preparing error data is done...");

}





void MultiLevelModel::initializeSurrogateModel(void){


	assert(ifBoxConstraintsAreSet);
	assert(ifSurrogateModelsAreSet);
	assert(ifDataIsRead);
	assert(ifNormalized);

	output.printMessage("Initializing the multi-level model...");
	output.printMessage("\nInitializing the low fidelity model...");

	lowFidelityModel->initializeSurrogateModel();
	output.printMessage("\nInitialization of the low fidelity model is done...");

	output.printMessage("\nInitializing the error model...");

	errorModel->initializeSurrogateModel();

	output.printMessage("Initialization of the error model is done...");

	ifInitialized = true;

	output.printMessage("Initialization of the multi-level model is done...");


}


void MultiLevelModel::printSurrogateModel(void) const{

	assert(ifInitialized);

	lowFidelityModel->printSurrogateModel();
	errorModel->printSurrogateModel();


}

void MultiLevelModel::printHyperParameters(void) const{





}

void MultiLevelModel::setWriteWarmStartFileFlag(bool flag){
	lowFidelityModel->setWriteWarmStartFileFlag(flag);
	errorModel->setWriteWarmStartFileFlag(flag);


}
void MultiLevelModel::setReadWarmStartFileFlag(bool flag){

	lowFidelityModel->setReadWarmStartFileFlag(flag);
	errorModel->setReadWarmStartFileFlag(flag);

}




void MultiLevelModel::saveHyperParameters(void) const{


	lowFidelityModel->saveHyperParameters();
	errorModel->saveHyperParameters();



}
void MultiLevelModel::loadHyperParameters(void){





}
void MultiLevelModel::updateAuxilliaryFields(void){

	lowFidelityModel->updateAuxilliaryFields();
	errorModel->updateAuxilliaryFields();



}


void MultiLevelModel::setNumberOfThreads(unsigned int nThreads){

	lowFidelityModel->setNumberOfThreads(nThreads);
	errorModel->setNumberOfThreads(nThreads);
	numberOfThreads = nThreads;


}

void MultiLevelModel::trainLowFidelityModel(void){

	assert(ifInitialized);

	output.printMessage("Training the low fidelity model...");
	lowFidelityModel->train();
}


void MultiLevelModel::trainErrorModel(void){

	assert(ifInitialized);

	output.printMessage("Training the error model...");
	errorModel->train();

}



void MultiLevelModel::train(void){

	output.printMessage("Training the multi-level model...");

	lowFidelityModel->train();
	errorModel->train();

}

void MultiLevelModel::determineAlpha(void){

	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	assert(numberOfSamplesLowFidelity>0);

	vec yHiFi = data.getOutputVector();
	vec yTemp = dataLowFidelity.getOutputVector();
	vec yLowFi(numberOfSamples, fill::zeros);

	for(unsigned int i=0; i<numberOfSamples; i++){
		unsigned int index = findIndexHiFiToLowFiData(i);
		yLowFi(i) = yTemp(index);
	}

	alpha = dot(yLowFi,yHiFi)/dot(yLowFi,yLowFi);
}


double MultiLevelModel::getAlpha(void) const{
	return alpha;
}



double MultiLevelModel::interpolate(rowvec x) const{

	double result = 0.0;

	double lowFidelityEstimate = lowFidelityModel->interpolate(x);
	double errorEstimate = errorModel->interpolate(x);
#if 0
	cout<<"lowFidelityEstimate ="<<lowFidelityEstimate<<"\n";
	cout<<"errorEstimate ="<<errorEstimate<<"\n";
	printScalar(alpha);
#endif

	result = alpha*lowFidelityEstimate + errorEstimate;
	return result;
}


double MultiLevelModel::interpolateLowFi(rowvec x) const{

	return lowFidelityModel->interpolate(x);

}

double MultiLevelModel::interpolateError(rowvec x) const{

	return errorModel->interpolate(x);

}


void MultiLevelModel::interpolateWithVariance(rowvec xp,double *estimatedValue,double *sigmaSquared) const{

	double lowFidelityEstimate;
	lowFidelityModel->interpolateWithVariance(xp,&lowFidelityEstimate,sigmaSquared);
	double errorEstimate = errorModel->interpolate(xp);

	*estimatedValue = alpha*lowFidelityEstimate +  errorEstimate;


}

unsigned int MultiLevelModel::findIndexHiFiToLowFiData(unsigned int indexHiFiData) const{

	assert(numberOfSamplesLowFidelity>0);
	assert(numberOfSamples>0);
	assert(numberOfSamplesLowFidelity >= numberOfSamples);
	assert(dimension>0);
	assert(indexHiFiData < numberOfSamples);

	unsigned int indexLoFi = 0;

	rowvec x(dimension);

	for(unsigned int i=0; i<dimension; i++) {

		x(i) = rawDataHighFidelity(indexHiFiData,i);
	}


	double minNorm = LARGE;
	for(unsigned int i=0; i < numberOfSamplesLowFidelity; i++){

		rowvec xp(dimension);
		for(unsigned int j=0; j<dimension; j++) xp(j) = rawDataLowFidelity(i,j);

		rowvec dx = x-xp;

		double normdx = calculateL1norm(dx);


		if(normdx < minNorm){

			minNorm = normdx;
			indexLoFi = i;
		}
	}

	if(minNorm > 10E-10){

		x.print("x = ");
		std::cout<<"Index in the high-fidelity training data = "<< indexHiFiData << "\n";

		abortWithErrorMessage("A high fidelity data point does not exist in the low fidelity data!");
	}


	return indexLoFi;

}



void MultiLevelModel::setIDHiFiModel(SURROGATE_MODEL id){

	modelIDHiFi = id;

}
void MultiLevelModel::setIDLowFiModel(SURROGATE_MODEL id){

	modelIDLowFi = id;
}




void MultiLevelModel::normalizeData(void){

	assert(ifBoxConstraintsAreSet);
	assert(ifSurrogateModelsAreSet);
	assert(ifDataIsRead);
	assert(ifErrorDataIsSet);

	data.normalize();
	dataLowFidelity.normalize();

	lowFidelityModel->normalizeData();
	errorModel->normalizeData();

	XLowFidelity = dataLowFidelity.getInputMatrix();
	XHighFidelity = data.getInputMatrix();


	ifNormalized = true;
}


void MultiLevelModel::bindModels(void){

	if(!checkifModelIDIsValid(modelIDHiFi)){
		abortWithErrorMessage("Undefined surrogate model for the high fidelity model");
	}
	if(!checkifModelIDIsValid(modelIDLowFi)){
		abortWithErrorMessage("Undefined surrogate model for the low fidelity model");
	}

	bindLowFidelityModel();
	bindErrorModel();

	ifSurrogateModelsAreSet = true;
}

bool MultiLevelModel::checkifModelIDIsValid(SURROGATE_MODEL id) const{

	if(id == NONE) return false;
	if(id == ORDINARY_KRIGING) return true;
	if(id == UNIVERSAL_KRIGING) return true;
	if(id == TANGENT) return true;
	if(id == GRADIENT_ENHANCED) return true;


	return false;
}

void MultiLevelModel::bindLowFidelityModel(void){

	assert(dimension>0);
	assert(isNotEmpty(filenameDataInputLowFidelity));

	if(!checkifModelIDIsValid(modelIDLowFi)){
		abortWithErrorMessage("Undefined surrogate model for the low fidelity model");
	}

	if(modelIDLowFi == ORDINARY_KRIGING){

		output.printMessage("Binding the low fidelity model with the Ordinary Kriging model...");
		lowFidelityModel = &surrogateModelKrigingLowFi;

	}

	if(modelIDLowFi == UNIVERSAL_KRIGING){

		output.printMessage("Binding the low fidelity model with the Universal Kriging model...");
		surrogateModelKrigingLowFi.setLinearRegressionOn();
		lowFidelityModel = &surrogateModelKrigingLowFi;

	}

	if(modelIDLowFi == TANGENT){

		output.printMessage("Binding the low fidelity model with the TEM model...");
		surrogateModelGGEKLowFi.setDirectionalDerivativesOn();
		lowFidelityModel = &surrogateModelGGEKLowFi;
		dataLowFidelity.setDirectionalDerivativesOn();

	}

	if(modelIDLowFi == GRADIENT_ENHANCED){

		output.printMessage("Binding the low fidelity model with the GGEK model...");
		lowFidelityModel = &surrogateModelGGEKLowFi;
		dataLowFidelity.setGradientsOn();

	}

	lowFidelityModel->setNameOfInputFile(filenameDataInputLowFidelity);
	lowFidelityModel->setDimension(dimension);

}

void MultiLevelModel::bindErrorModel(void){

	if(modelIDLowFi == ORDINARY_KRIGING && modelIDHiFi == ORDINARY_KRIGING){

		output.printMessage("Binding the error model with the Ordinary Kriging model...");
		errorModel = &surrogateModelKrigingError;
		modelIDError = ORDINARY_KRIGING;

	}

	if((modelIDLowFi == TANGENT && modelIDHiFi == ORDINARY_KRIGING)
	|| (modelIDLowFi == GRADIENT_ENHANCED && modelIDHiFi == ORDINARY_KRIGING)){

		output.printMessage("Binding the error model with the Ordinary Kriging model...");
		errorModel = &surrogateModelKrigingError;

		modelIDError = ORDINARY_KRIGING;

	}


	if(modelIDLowFi == TANGENT && modelIDHiFi == TANGENT){

		output.printMessage("Binding the error model with the TEM model...");
		assert(false);

	}

	errorModel->setDimension(dimension);

}





mat MultiLevelModel::getRawDataHighFidelity(void) const{
	assert(numberOfSamples>0);
	return rawDataHighFidelity;

}
mat MultiLevelModel::getRawDataLowFidelity(void) const{

	assert(numberOfSamplesLowFidelity>0);
	return rawDataLowFidelity;
}

mat MultiLevelModel::getRawDataError(void) const{

	assert(rawDataError.n_rows > 0 );
	return rawDataError;
}

unsigned int MultiLevelModel::getNumberOfLowFiSamples(void) const{
	return numberOfSamplesLowFidelity;
}
unsigned int MultiLevelModel::getNumberOfHiFiSamples(void) const{
	return numberOfSamples;
}

void MultiLevelModel::addNewSampleToData(rowvec newsample){

	assert(isNotEmpty(filenameDataInput));
	appendRowVectorToCSVData(newsample, filenameDataInput);
	readData();
	normalizeData();
	initializeSurrogateModel();

}


void MultiLevelModel::addNewLowFidelitySampleToData(rowvec newsample){

	assert(isNotEmpty(filenameDataInputLowFidelity));
	appendRowVectorToCSVData(newsample, filenameDataInputLowFidelity);
	readData();
	normalizeData();

	initializeSurrogateModel();

}

