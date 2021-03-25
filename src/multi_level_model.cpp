/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */

#include <armadillo>
#include<cassert>
#include "multi_level_method.hpp"
#include "kriging_training.hpp"
#include "gek.hpp"
#include "aggregation_model.hpp"
#include "matrix_vector_operations.hpp"
#include "Rodeo_globals.hpp"
#include "metric.hpp"
using namespace arma;


MultiLevelModel::MultiLevelModel(std::string label)
:SurrogateModel(label) {

	inputFileNameError = label+"_Error.csv";


}



void MultiLevelModel::initializeSurrogateModel(void){

	assert(ifLowFidelityModelIsSet);
	assert(ifErrorModelIsSet);

	readData();

	lowFidelityModel->setParameterBounds(xmin,xmax);
	errorModel->setParameterBounds(xmin,xmax);
	lowFidelityModel->normalizeData();
	errorModel->normalizeData();

	lowFidelityModel->initializeSurrogateModel();
	errorModel->initializeSurrogateModel();


}


void MultiLevelModel::printSurrogateModel(void) const{

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

void MultiLevelModel::train(void){


	lowFidelityModel->train();
	errorModel->train();



}

double evaluateBrigdeFunction(rowvec x){

	double brigdgeFunctionValue = 1.0;

	return brigdgeFunctionValue;


}

double MultiLevelModel::interpolate(rowvec x) const{

	double result = 0.0;

	double lowFidelityEstimate = lowFidelityModel->interpolate(x);
	double errorEstimate = errorModel->interpolate(x);
#if 0
	std::cout<<"lowFidelityEstimate ="<<lowFidelityEstimate<<"\n";
	std::cout<<"errorEstimate ="<<errorEstimate<<"\n";
#endif

	result = lowFidelityEstimate + errorEstimate;

	return result;

}
void MultiLevelModel::interpolateWithVariance(rowvec xp,double *estimatedValue,double *sigmaSquared) const{

	double lowFidelityEstimate;
	lowFidelityModel->interpolateWithVariance(xp,&lowFidelityEstimate,sigmaSquared);
	double errorEstimate = errorModel->interpolate(xp);

	*estimatedValue = lowFidelityEstimate + errorEstimate;


}

void MultiLevelModel::setinputFileNameHighFidelityData(std::string filename){

	assert(!filename.empty());
	inputFileNameHighFidelityData = filename;

}

void MultiLevelModel::setinputFileNameLowFidelityData(std::string filename){

	assert(!filename.empty());
	inputFileNameLowFidelityData = filename;

}

void MultiLevelModel::readHighFidelityData(void){

	rawDataHighFidelity = readMatFromCVSFile(inputFileNameHighFidelityData);

	NHiFi =  rawDataHighFidelity.n_rows;

}
void MultiLevelModel::readLowFidelityData(void){

	rawDataLowFidelity = readMatFromCVSFile(inputFileNameLowFidelityData);

	NLoFi =  rawDataLowFidelity.n_rows;

}

unsigned int MultiLevelModel::findIndexHiFiToLowFiData(unsigned int indexHiFiData) const{

	assert(NLoFi>0);
	assert(NHiFi>0);
	assert(NLoFi >= NHiFi);
	assert(dimHiFi >0);
	assert(dimLoFi >0);
	assert(indexHiFiData < NHiFi);

	unsigned int indexLoFi = 0;

	rowvec x(dimHiFi);

	for(unsigned int i=0; i<dimHiFi; i++) x(i) = rawDataHighFidelity(indexHiFiData,i);

	double minNorm = LARGE;
	for(unsigned int i=0; i < NLoFi; i++){

		rowvec xp(dimLoFi);
		for(unsigned int j=0; j<dimLoFi; j++) xp(j) = rawDataLowFidelity(i,j);

		rowvec dx = x-xp;

		double normdx = calculateL1norm(dx);


		if(normdx <minNorm){


			minNorm = normdx;
			indexLoFi = i;
		}

	}

	if(minNorm > 10E-10){

		std::cout<<"ERROR (Multilevel model): A high fidelity data point does not exist in the low fidelity data!\n";
		abort();
	}


	return indexLoFi;

}

void MultiLevelModel::prepareErrorData(void){

	assert(NLoFi>0);
	assert(NHiFi>0);
	assert(NLoFi >= NHiFi);

	/* if high fidelity data has gradients, low fidelity must also have gradients */
	if(ifHighFidelityDataHasGradients) assert(ifLowFidelityDataHasGradients);

	if(ifHighFidelityDataHasGradients){

		rawDataError = zeros<mat>(NHiFi,2*dimHiFi+1);

	}
	else{

		rawDataError = zeros<mat>(NHiFi,dimHiFi+1);

	}

	for(unsigned int i = 0; i < NHiFi; i++){


		unsigned int indexLowFi = findIndexHiFiToLowFiData(i);
		double error = rawDataHighFidelity(i,dimHiFi) - rawDataLowFidelity(indexLowFi,dimLoFi);

		for(unsigned int j = 0; j < dimHiFi; j++){

			rawDataError(i,j) = rawDataHighFidelity(i,j);

		}
		rawDataError(i,dimHiFi) = error;

		if(ifHighFidelityDataHasGradients){

			for(unsigned int j = 0; j < NHiFi; j++){

				double errorGradient = rawDataHighFidelity(i,dimHiFi+1+j) - rawDataHighFidelity(indexLowFi,dimLoFi+1+j);

				rawDataError(i,dimHiFi+i+j) = errorGradient;
			}

		}


	}


	rawDataError.save(inputFileNameError, csv_ascii);


	ifErrorDataIsSet = true;
}


void MultiLevelModel::setDimensionsHiFiandLowFiModels(void){

	unsigned int nColsLoFiData = this->rawDataLowFidelity.n_cols;
	unsigned int nColsHiFiData = this->rawDataHighFidelity.n_cols;


	if(this->ifLowFidelityDataHasGradients){

		dimLoFi = (nColsLoFiData-1)/2;

	}
	else{

		dimLoFi = nColsLoFiData-1;
	}

	if(this->ifHighFidelityDataHasGradients){

		dimHiFi = (nColsHiFiData-1)/2;

	}
	else{

		dimHiFi = nColsHiFiData-1;
	}


	assert(dimHiFi == dimLoFi);

}

void MultiLevelModel::readData(void){

	assert(ifLowFidelityModelIsSet);
	assert(ifErrorModelIsSet);


	readHighFidelityData();
	readLowFidelityData();

	setDimensionsHiFiandLowFiModels();

	prepareErrorData();

	lowFidelityModel->setNameOfInputFile(inputFileNameLowFidelityData);
	lowFidelityModel->readData();

	errorModel->setNameOfInputFile(inputFileNameError);
	errorModel->readData();

	ifDataIsRead = true;


}

void MultiLevelModel::setLowFidelityModel(std::string surrogateModelType){

	assert(ifModelIsValid(surrogateModelType));
	std::string labelLowFiModel = label + "_LowFi";
	KrigingModel* LowFiModelKriging = new KrigingModel(labelLowFiModel);
	LinearModel* LowFiModelLinearRegression = new LinearModel(labelLowFiModel);
	AggregationModel* LowFiModelAggregation = new AggregationModel(labelLowFiModel);
	GEKModel* LowFiModelGEK = new GEKModel(labelLowFiModel);

	if(surrogateModelType == "ORDINARY_KRIGING"){
		lowFidelityModel =  LowFiModelKriging;

	}
	if(surrogateModelType == "UNIVERSAL_KRIGING"){
		LowFiModelKriging->setLinearRegressionOn();
		lowFidelityModel =  LowFiModelKriging;
	}
	if(surrogateModelType == "LINEAR_REGRESSION"){
		lowFidelityModel = LowFiModelLinearRegression;
	}
	if(surrogateModelType == "GRADIENT_ENHANCED_KRIGING"){
		lowFidelityModel = LowFiModelGEK;
		ifLowFidelityDataHasGradients = true;
	}

	if(surrogateModelType == "AGGREGATION"){
		lowFidelityModel = LowFiModelAggregation;
		ifLowFidelityDataHasGradients = true;
	}


	ifLowFidelityModelIsSet = true;

}

void MultiLevelModel::setErrorModel(std::string surrogateModelType){


	assert(ifModelIsValid(surrogateModelType));
	std::string labelErrorModel = label + "_Error";
	KrigingModel* errorModelKriging = new KrigingModel(labelErrorModel );
	LinearModel* errorModelLinearRegression = new LinearModel(labelErrorModel );
	AggregationModel* errorModelAggregation = new AggregationModel(labelErrorModel );
	GEKModel *errorModelGEK = new GEKModel(labelErrorModel );

	if(surrogateModelType == "ORDINARY_KRIGING"){
		errorModel =  errorModelKriging;
	}
	if(surrogateModelType == "UNIVERSAL_KRIGING"){
		errorModelKriging->setLinearRegressionOn();
		errorModel =  errorModelKriging;
	}
	if(surrogateModelType == "LINEAR_REGRESSION"){
		errorModel = errorModelLinearRegression;
	}
	if(surrogateModelType == "GRADIENT_ENHANCED_KRIGING"){
		errorModel = errorModelGEK;
		ifHighFidelityDataHasGradients = true;
	}

	if(surrogateModelType == "AGGREGATION"){
		errorModel = errorModelAggregation;
		ifHighFidelityDataHasGradients = true;
	}


	ifErrorModelIsSet = true;

}


void MultiLevelModel::setParameterBounds(vec lb, vec ub){



	xmin = lb;
	xmax = ub;
	checkIfParameterBoundsAreOk();
	ifBoundsAreSet = true;


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

