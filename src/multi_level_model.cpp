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

//MultiLevelModel::MultiLevelModel():SurrogateModel(){}

void MultiLevelModel::setNameOfInputFile(string filename){}

void MultiLevelModel::setNameOfInputFileError(void){

	assert(isNotEmpty(name));
	inputFileNameError = name +"_Error.csv";
	output.printMessage("Error data is set as ", inputFileNameError);

}


void MultiLevelModel::setNameOfHyperParametersFile(string label){

	assert(isNotEmpty(label));

	hyperparameters_filename = label + "multilevel_hyperparameters.csv";


}
void MultiLevelModel::setNumberOfTrainingIterations(unsigned int nIter){

	lowFidelityModel->setNumberOfTrainingIterations(nIter);
	errorModel->setNumberOfTrainingIterations(nIter);

	output.printMessage("Number of training iterations is set as" + numberOfTrainingIterations);

}

void MultiLevelModel::setinputFileNameHighFidelityData(string filename){

	assert(isNotEmpty(filename));

	inputFileNameHighFidelityData = filename;
	filenameDataInput = filename;

	output.printMessage("Name for the high fidelity input data is set as " + inputFileNameHighFidelityData);
}

void MultiLevelModel::setinputFileNameLowFidelityData(string filename){

	assert(isNotEmpty(filename));

	inputFileNameLowFidelityData = filename;
	output.printMessage("Name for the low fidelity input data is set as " + inputFileNameLowFidelityData);

}


void MultiLevelModel::setDisplayOn(void){
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

	boxConstraints = boxConstraintsInput;

	lowFidelityModel->setBoxConstraints(boxConstraintsInput);
	errorModel->setBoxConstraints(boxConstraintsInput);

	ifBoxConstraintsAreSet = true;

}


void MultiLevelModel::initializeSurrogateModel(void){

	assert(ifBoxConstraintsAreSet);
	assert(ifSurrogateModelsAreSet);
	assert(ifDataIsRead);

	output.printMessage("Initializing the multi-level model...");
	output.printMessage("\nInitializing the low fidelity model...");

	lowFidelityModel->setNameOfInputFile(inputFileNameLowFidelityData);
	lowFidelityModel->readData();
	lowFidelityModel->normalizeData();
	lowFidelityModel->initializeSurrogateModel();


	output.printMessage("\nInitialization of the low fidelity model is done...");


	output.printMessage("\nInitializing the error model...");

	prepareErrorData();

	errorModel->setNameOfInputFile(inputFileNameError);
	errorModel->readData();
	errorModel->normalizeData();
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

	unsigned int numberOfHiFiSamplesAuxiliaryModel = NHiFi/5;

	assert(numberOfHiFiSamplesAuxiliaryModel > 0);

	mat rawDataAuxiliaryHiFiSamples =  rawDataHighFidelity;
	rawDataAuxiliaryHiFiSamples = shuffle(rawDataAuxiliaryHiFiSamples);


	unsigned int nCols = rawDataAuxiliaryHiFiSamples.n_cols;

	mat auxiliaryHiFiSamplesTest     = rawDataAuxiliaryHiFiSamples.submat(0,0,numberOfHiFiSamplesAuxiliaryModel-1, nCols-1);
	mat auxiliaryHiFiSamplesTraining = rawDataAuxiliaryHiFiSamples.submat(numberOfHiFiSamplesAuxiliaryModel,0,NHiFi-1, nCols-1);

	saveMatToCVSFile(auxiliaryHiFiSamplesTraining,"auxiliaryHiFiSamplesTraining.csv");

	auxiliaryModel.setName("auxiliaryModelForGammaTraining");
	auxiliaryModel.setinputFileNameHighFidelityData("auxiliaryHiFiSamplesTraining.csv");
	auxiliaryModel.setinputFileNameLowFidelityData(inputFileNameLowFidelityData);



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


void MultiLevelModel::readHighFidelityData(void){

	assert(isNotEmpty(inputFileNameHighFidelityData));

	rawDataHighFidelity = readMatFromCVSFile(inputFileNameHighFidelityData);
	NHiFi =  rawDataHighFidelity.n_rows;

}
void MultiLevelModel::readLowFidelityData(void){

	assert(isNotEmpty(inputFileNameLowFidelityData));

	rawDataLowFidelity = readMatFromCVSFile(inputFileNameLowFidelityData);
	NLoFi =  rawDataLowFidelity.n_rows;
	lowFidelityModel->readData();

}

unsigned int MultiLevelModel::findIndexHiFiToLowFiData(unsigned int indexHiFiData) const{

	assert(NLoFi>0);
	assert(NHiFi>0);
	assert(NLoFi >= NHiFi);
	assert(dimension>0);
	assert(indexHiFiData < NHiFi);

	unsigned int indexLoFi = 0;

	rowvec x(dimension);

	for(unsigned int i=0; i<dimension; i++) x(i) = rawDataHighFidelity(indexHiFiData,i);

	double minNorm = LARGE;
	for(unsigned int i=0; i < NLoFi; i++){

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

void MultiLevelModel::prepareErrorData(void){

	assert(dimension>0);
	assert(NHiFi>0);
	assert(NLoFi>0);
	assert(NLoFi >= NHiFi);

	output.printMessage("Preparing error data...");

	setNameOfInputFileError();

	if(modelIDError == ORDINARY_KRIGING){
		rawDataError = zeros<mat>(NHiFi,dimension+1);
	}

	for(unsigned int i = 0; i < NHiFi; i++){

		unsigned int indexLowFi = findIndexHiFiToLowFiData(i);

		double error = rawDataHighFidelity(i,dimension) - rawDataLowFidelity(indexLowFi,dimension);

		for(unsigned int j = 0; j < dimension; j++){
			rawDataError(i,j) = rawDataHighFidelity(i,j);
		}
		rawDataError(i,dimension) = error;

	}




	//
	//
	//	/* if high fidelity data has gradients, low fidelity must also have gradients */
	//	if(ifHighFidelityDataHasGradients) assert(ifLowFidelityDataHasGradients);
	//
	//	if(ifHighFidelityDataHasGradients){
	//
	//		rawDataError = zeros<mat>(NHiFi,2*dimHiFi+1);
	//
	//	}
	//	else{
	//
	//		rawDataError = zeros<mat>(NHiFi,dimHiFi+1);
	//
	//	}
	//

		/* difference in gradients */

		//			if(ifHighFidelityDataHasGradients){
		//
		//				for(unsigned int j = 0; j < dimHiFi; j++){
		//
		//					double errorGradient = rawDataHighFidelity(i,dimHiFi+1+j) - rawDataHighFidelity(indexLowFi,dimLoFi+1+j);
		//
		//					rawDataError(i,dimHiFi+1+j) = errorGradient;
		//				}
		//
		//			}
//	}
	//
	assert(isNotEmpty(inputFileNameError));

	rawDataError.save(inputFileNameError, csv_ascii);

	output.printMessage("Error data is saved in " + inputFileNameError);

	ifErrorDataIsSet = true;

	output.printMessage("Preparing error data is done...");

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

void MultiLevelModel::readData(void){

	assert(dimension>0);
	assert(ifSurrogateModelsAreSet);

	output.printMessage("Reading data for the multilevel model...");

	data.readData(filenameDataInput);

	readHighFidelityData();
	readLowFidelityData();


	XLowFidelity =  rawDataLowFidelity.submat(0,0,NLoFi -1,  dimension -1);
	XHighFidelity = rawDataHighFidelity.submat(0,0,NHiFi -1, dimension -1);


	//	prepareErrorData();

	ifDataIsRead = true;
	output.printMessage("Reading data for the multilevel model is done...");




}

void MultiLevelModel::normalizeData(void){

	XLowFidelity  = normalizeMatrix(XLowFidelity, boxConstraints);
	XLowFidelity = (1.0/dimension)*XLowFidelity;
	XHighFidelity  = normalizeMatrix(XHighFidelity, boxConstraints);
	XHighFidelity = (1.0/dimension)*XHighFidelity;

	ifNormalized = true;
}


void MultiLevelModel::bindModels(void){

	assert(isNotEmpty(inputFileNameHighFidelityData));
	assert(isNotEmpty(inputFileNameLowFidelityData));

	bindLowFidelityModel();
	bindErrorModel();

	ifSurrogateModelsAreSet = true;
}

bool MultiLevelModel::checkifModelIDIsValid(SURROGATE_MODEL id) const{

	if(id == NONE) return false;
	if(id == ORDINARY_KRIGING) return true;
	if(id == UNIVERSAL_KRIGING) return true;


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

	lowFidelityModel->setNameOfInputFile(inputFileNameLowFidelityData);

}

void MultiLevelModel::bindErrorModel(void){

	assert(isNotEmpty(inputFileNameError));

	if(!checkifModelIDIsValid(modelIDHiFi)){
		abortWithErrorMessage("Undefined surrogate model for the high fidelity model");
	}

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









}





mat MultiLevelModel::getRawDataHighFidelity(void) const{

	assert(NHiFi>0);
	return rawDataHighFidelity;

}
mat MultiLevelModel::getRawDataLowFidelity(void) const{

	assert(NLoFi>0);
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
	return NLoFi;
}
unsigned int MultiLevelModel::getNumberOfHiFiSamples(void) const{
	return NHiFi;
}

void MultiLevelModel::addNewSampleToData(rowvec newsample){


	assert(isNotEmpty(inputFileNameHighFidelityData));
	appendRowVectorToCSVData(newsample, inputFileNameHighFidelityData);
	initializeSurrogateModel();



}


void MultiLevelModel::addNewLowFidelitySampleToData(rowvec newsample){


	assert(isNotEmpty(inputFileNameLowFidelityData));
	appendRowVectorToCSVData(newsample, inputFileNameLowFidelityData);
	initializeSurrogateModel();

}

