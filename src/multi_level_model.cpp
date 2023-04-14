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
#include "multi_level_method.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "matrix_vector_operations.hpp"
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

	assert(ifSurrogateModelsAreSet);
	dimension = dim;
	lowFidelityModel->setDimension(dim);
	errorModel->setDimension(dim);
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

	hyperparameters_filename = label;


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

void  MultiLevelModel::setGamma(double value){
	gamma = value;
}
double  MultiLevelModel::getGamma(void) const{
	return gamma;
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

	setNameOfInputFileError();

	if(modelIDError == ORDINARY_KRIGING){
		rawDataError = zeros<mat>(numberOfSamples,dimension+1);
	}
	if(modelIDError == TANGENT){
		rawDataError = zeros<mat>(numberOfSamples,2*dimension+2);
	}

	for(unsigned int i = 0; i < numberOfSamples; i++){

		unsigned int indexLowFi = findIndexHiFiToLowFiData(i);

		double error = rawDataHighFidelity(i,dimension) - rawDataLowFidelity(indexLowFi,dimension);

		for(unsigned int j = 0; j < dimension; j++){
			rawDataError(i,j) = rawDataHighFidelity(i,j);
		}
		rawDataError(i,dimension) = error;

	}

	if(modelIDError == TANGENT){

		for(unsigned int i = 0; i < numberOfSamples; i++){

			unsigned int indexLowFi = findIndexHiFiToLowFiData(i);

			double errorDirectionalDerivative = rawDataHighFidelity(i,dimension+1) - rawDataLowFidelity(indexLowFi,dimension+1);

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
void MultiLevelModel::saveHyperParameters(void) const{





}
void MultiLevelModel::loadHyperParameters(void){





}
void MultiLevelModel::updateAuxilliaryFields(void){

	lowFidelityModel->updateAuxilliaryFields();
	errorModel->updateAuxilliaryFields();



}

void MultiLevelModel::setNumberOfMaximumIterationsForGammaTraining(unsigned int value){

	maxIterationsForGammaTraining = value;


}


void MultiLevelModel::setNumberOfThreads(unsigned int nThreads){

	lowFidelityModel->setNumberOfThreads(nThreads);
	errorModel->setNumberOfThreads(nThreads);
	numberOfThreads = nThreads;


}

void MultiLevelModel::determineGammaBasedOnData(void){

	assert(ifDataIsRead);
	assert(ifBoxConstraintsAreSet);



	MultiLevelModel auxiliaryModel;

	/* we just copy hyperparameters in this way to avoid model training */
	auxiliaryModel = *this;

	unsigned int numberOfHiFiSamplesAuxiliaryModel = numberOfSamples/5;

	assert(numberOfHiFiSamplesAuxiliaryModel > 0);

	mat rawDataAuxiliaryHiFiSamples =  rawDataHighFidelity;
	rawDataAuxiliaryHiFiSamples = shuffle(rawDataAuxiliaryHiFiSamples);


	unsigned int nCols = rawDataAuxiliaryHiFiSamples.n_cols;

	mat auxiliaryHiFiSamplesTest     = rawDataAuxiliaryHiFiSamples.submat(0,0,numberOfHiFiSamplesAuxiliaryModel-1, nCols-1);
	mat auxiliaryHiFiSamplesTraining = rawDataAuxiliaryHiFiSamples.submat(numberOfHiFiSamplesAuxiliaryModel,0,numberOfSamples-1, nCols-1);

	saveMatToCVSFile(auxiliaryHiFiSamplesTraining,"auxiliaryHiFiSamplesTraining.csv");

	auxiliaryModel.setName("auxiliaryModelForGammaTraining");
	auxiliaryModel.setinputFileNameHighFidelityData("auxiliaryHiFiSamplesTraining.csv");
	auxiliaryModel.setinputFileNameLowFidelityData(filenameDataInputLowFidelity);



	auxiliaryModel.setBoxConstraints(boxConstraints);

	auxiliaryModel.initializeSurrogateModel();

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();

	mat XTest = auxiliaryHiFiSamplesTest.submat(0, 0, numberOfHiFiSamplesAuxiliaryModel-1,  nCols-2);
	vec yTest = auxiliaryHiFiSamplesTest.col(nCols-1);



	double gammaMax = 10.0;
	double gammaMin =  0.0;
	double deltaGamma = (gammaMax - gammaMin)/ maxIterationsForGammaTraining;

	double gammaToSet = 0.0;
	double bestGamma = 0.0;
	double bestSE = LARGE;

	for(unsigned int iterGamma=0; iterGamma<maxIterationsForGammaTraining; ++iterGamma){

		auxiliaryModel.setGamma(gammaToSet);
		double SE = 0.0;
		for(unsigned int i=0; i<numberOfHiFiSamplesAuxiliaryModel; ++i){

			rowvec x = XTest.row(i);
			rowvec xNormalized = normalizeRowVector(x,lb,ub);

			double fTilde = auxiliaryModel.interpolate(xNormalized);
			double f = yTest(i);

			SE += ( f - fTilde) * ( f - fTilde);

		}

		if(SE < bestSE){

			bestSE = SE;
			bestGamma = gammaToSet;

		}


		gammaToSet +=deltaGamma;

	}

	output.printMessage("Gamma training is done...");
	output.printMessage("Best value for gamma = " , bestGamma );
	setGamma(bestGamma);

}




void MultiLevelModel::trainLowFidelityModel(void){

	assert(ifInitialized);

	output.printMessage("Training the low fidelity model...");
	lowFidelityModel->train();
	lowFidelityModel->setNameOfHyperParametersFile("lowFidelityModel");

}


void MultiLevelModel::trainErrorModel(void){

	assert(ifInitialized);

	output.printMessage("Training the error model...");
	errorModel->train();
	errorModel->setNameOfHyperParametersFile("errorModel");
	errorModel->saveHyperParameters();

}



void MultiLevelModel::train(void){

	output.printMessage("Training the multi-level model...");

	lowFidelityModel->train();
	errorModel->train();

	determineGammaBasedOnData();
}






double MultiLevelModel::interpolate(rowvec x) const{

	double result = 0.0;
	double alpha = 1.0;

	double lowFidelityEstimate = lowFidelityModel->interpolate(x);
	double errorEstimate = errorModel->interpolate(x);
#if 0
	cout<<"lowFidelityEstimate ="<<lowFidelityEstimate<<"\n";
	cout<<"errorEstimate ="<<errorEstimate<<"\n";
#endif


	double distanceToHF = findNearestL1DistanceToAHighFidelitySample(x);
	double distanceToLF = findNearestL1DistanceToALowFidelitySample(x);

#if 0
	cout<<"distToHF ="<<distanceToHF<<"\n";
	cout<<"distToLF ="<<distanceToLF<<"\n";

#endif


	if(distanceToLF < distanceToHF){

		alpha = exp(-gamma*distanceToHF);
#if 0
		cout<<"alpha = "<<alpha<<"\n";
		cout<<"gamma = "<<gamma<<"\n";
#endif


	}


	result = lowFidelityEstimate + alpha*errorEstimate;
	double resultWithoutAlpha = lowFidelityEstimate + errorEstimate;

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


	double distanceToHF = findNearestL1DistanceToAHighFidelitySample(xp);
	double distanceToLF = findNearestL1DistanceToALowFidelitySample(xp);

#if 0
	cout<<"distToHF ="<<distanceToHF<<"\n";
	cout<<"distToLF ="<<distanceToLF<<"\n";

#endif

	double alpha = 1.0;

	if(distanceToLF < distanceToHF){

		alpha = exp(-gamma*distanceToHF);
#if 0
		cout<<"alpha = "<<alpha<<"\n";
		cout<<"gamma = "<<gamma<<"\n";
#endif


	}


	*estimatedValue = lowFidelityEstimate + alpha * errorEstimate;


}

unsigned int MultiLevelModel::findIndexHiFiToLowFiData(unsigned int indexHiFiData) const{

	assert(numberOfSamplesLowFidelity>0);
	assert(numberOfSamples>0);
	assert(numberOfSamplesLowFidelity >= numberOfSamples);
	assert(dimension>0);
	assert(indexHiFiData < numberOfSamples);

	unsigned int indexLoFi = 0;

	rowvec x(dimension);

	for(unsigned int i=0; i<dimension; i++) x(i) = rawDataHighFidelity(indexHiFiData,i);

	double minNorm = LARGE;
	for(unsigned int i=0; i < numberOfSamplesLowFidelity; i++){

		rowvec xp(dimension);
		for(unsigned int j=0; j<dimension; j++) xp(j) = rawDataLowFidelity(i,j);

		rowvec dx = x-xp;

		double normdx = calculateL1norm(dx);


		if(normdx <minNorm){

			minNorm = normdx;
			indexLoFi = i;
		}
	}

	if(minNorm > 10E-10){
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




void MultiLevelModel::setDimensionsHiFiandLowFiModels(void){

	//	unsigned int nColsLoFiData = this->rawDataLowFidelity.n_cols;
	//	unsigned int nColsHiFiData = this->rawDataHighFidelity.n_cols;
	//
	//
	//	if(this->ifLowFidelityDataHasGradients){
	//
	//		dimLoFi = (nColsLoFiData-1)/2;
	//
	//	}
	//	else{
	//
	//		dimLoFi = nColsLoFiData-1;
	//	}
	//
	//	if(this->ifHighFidelityDataHasGradients){
	//
	//		dimHiFi = (nColsHiFiData-1)/2;
	//
	//	}
	//	else{
	//
	//		dimHiFi = nColsHiFiData-1;
	//	}
	//
	//
	//	if(dimHiFi != dimLoFi){
	//
	//		cout<<"ERROR: dimHiFi != dimLoFi!\n";
	//		cout<<"dimHiFi = "<<dimHiFi<<"\n";
	//		cout<<"dimLoFi = "<<dimLoFi<<"\n";
	//		abort();
	//
	//	}
	//
	//	data.setDimension(dimHiFi);

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


	return false;
}

void MultiLevelModel::bindLowFidelityModel(void){

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
		lowFidelityModel = &surrogateModelTGEKLowFi;
		dataLowFidelity.setDirectionalDerivativesOn();

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

	if(modelIDLowFi == TANGENT && modelIDHiFi == ORDINARY_KRIGING){

		output.printMessage("Binding the error model with the Ordinary Kriging model...");
		errorModel = &surrogateModelKrigingError;

		modelIDError = ORDINARY_KRIGING;

	}


	if(modelIDLowFi == TANGENT && modelIDHiFi == TANGENT){

		output.printMessage("Binding the error model with the TEM model...");
		errorModel = &surrogateModelTEMError;

		modelIDError = TANGENT;

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



unsigned int MultiLevelModel::findNearestNeighbourLowFidelity(rowvec x) const{

	return findNearestNeighborL1(x, XLowFidelity);

}

unsigned int MultiLevelModel::findNearestNeighbourHighFidelity(rowvec x) const{

	return findNearestNeighborL1(x, XHighFidelity);


}


double MultiLevelModel::findNearestL1DistanceToALowFidelitySample(rowvec x) const{

	unsigned int indx =  findNearestNeighborL1(x, XLowFidelity);

	rowvec xp = XLowFidelity.row(indx);
	rowvec diff = x- xp;

	return calculateL1norm(diff);



}

double MultiLevelModel::findNearestL1DistanceToAHighFidelitySample(rowvec x) const{

	unsigned int indx =  findNearestNeighborL1(x, XHighFidelity);

	rowvec xp = XHighFidelity.row(indx);
	rowvec diff = x- xp;

	return calculateL1norm(diff);

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

