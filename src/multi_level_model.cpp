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


MultiLevelModel::MultiLevelModel(){


}


MultiLevelModel::MultiLevelModel(std::string label){

	inputFileNameError = label+"_Error.csv";
	hyperparameters_filename = label + "_multilevel_model_hyperparameters.csv";


}


void MultiLevelModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());


}
void MultiLevelModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());

}
void MultiLevelModel::setNumberOfTrainingIterations(unsigned int nIter){

	lowFidelityModel->setNumberOfTrainingIterations(nIter);
	errorModel->setNumberOfTrainingIterations(nIter);

	if(ifDisplay){

		std::cout<<"numberOfTrainingIterations is set as "<<numberOfTrainingIterations<<" for the multi-level model\n";

	}

}

void MultiLevelModel::setinputFileNameHighFidelityData(std::string filename){

	assert(!filename.empty());
	inputFileNameHighFidelityData = filename;
	ifInputFileNameForHiFiModelIsSet = true;


}

void MultiLevelModel::setinputFileNameLowFidelityData(std::string filename){

	assert(!filename.empty());
	inputFileNameLowFidelityData = filename;
	ifInputFileNameForLowFiModelIsSet = true;

}



void MultiLevelModel::initializeSurrogateModel(void){

	assert(ifBoundsAreSet);


	bindErrorModel();
	bindLowFidelityModel();

	readData();


	lowFidelityModel->setNameOfInputFile(inputFileNameLowFidelityData);

	lowFidelityModel->readData();
	lowFidelityModel->setParameterBounds(xmin,xmax);
	lowFidelityModel->normalizeData();
	lowFidelityModel->initializeSurrogateModel();



	errorModel->setNameOfInputFile(inputFileNameError);

	errorModel->readData();
	errorModel->setParameterBounds(xmin,xmax);
	errorModel->normalizeData();
	errorModel->initializeSurrogateModel();

	ifInitialized = true;


	if(ifDisplay){

		printSurrogateModel();

	}


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


void MultiLevelModel::determineGammaBasedOnData(void){

	printMsg("Determining the variable gamma...");

	vec dist(100);
	for(int i=0; i<100; i++){

		rowvec x(dim);
		x = generateRandomRowVector(xmin,xmax);
		x = normalizeRowVector(x, xmin, xmax);

		x.print();

		dist(i) = findNearestL1DistanceToAHighFidelitySample(x);

	}

	double distMax = max(dist);

	if(ifDisplay){

		std::cout<<"distMax = "<<distMax<<"\n";

	}

	gamma = 0.1/exp(-distMax);

	/* alpha = gamma * exp(-distMax) = 0.1
	 *
	 *
	 *
	 * */

	if(ifDisplay){
		std::cout<<"gamma = "<<gamma<<"\n";
		std::cout<<"exp(-distMax) = "<<exp(-distMax)<<"\n";
		std::cout<<"alpha at distMax = "<<gamma* exp(-distMax)<<"\n";

	}




}




void MultiLevelModel::trainLowFidelityModel(void){

	assert(ifInitialized);

	if(ifDisplay){

		lowFidelityModel->ifDisplay= true;
	}

	lowFidelityModel->train();

}


void MultiLevelModel::trainErrorModel(void){

	assert(ifInitialized);

	if(ifDisplay){

		errorModel->ifDisplay= true;
	}

	errorModel->train();

}



void MultiLevelModel::train(void){


	lowFidelityModel->train();
	errorModel->train();


	determineGammaBasedOnData();


}


double MultiLevelModel::interpolate(rowvec x) const{

	double result = 0.0;
	double alpha = 1.0;

	double lowFidelityEstimate = lowFidelityModel->interpolate(x);
	double errorEstimate = errorModel->interpolate(x);
#if 1
	std::cout<<"lowFidelityEstimate ="<<lowFidelityEstimate<<"\n";
	std::cout<<"errorEstimate ="<<errorEstimate<<"\n";
#endif


	double distToHF = findNearestL1DistanceToAHighFidelitySample(x);
	double distToLF = findNearestL1DistanceToALowFidelitySample(x);

#if 1
	std::cout<<"distToHF ="<<distToHF<<"\n";
	std::cout<<"distToLF ="<<distToLF<<"\n";

#endif


	if(distToLF < distToHF){

		alpha = gamma*exp(-distToHF);
#if 1
		std::cout<<"alpha = "<<alpha<<"\n";
#endif


	}


	result = lowFidelityEstimate + alpha*errorEstimate;

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

	*estimatedValue = lowFidelityEstimate + errorEstimate;


}


void MultiLevelModel::readHighFidelityData(void){

	assert(ifInputFileNameForHiFiModelIsSet);
	rawDataHighFidelity = readMatFromCVSFile(inputFileNameHighFidelityData);

	NHiFi =  rawDataHighFidelity.n_rows;

}
void MultiLevelModel::readLowFidelityData(void){

	assert(ifInputFileNameForLowFiModelIsSet);
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

	printMsg("Preparing error data...");

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

		/* difference in gradients */

		if(ifHighFidelityDataHasGradients){

			for(unsigned int j = 0; j < dimHiFi; j++){

				double errorGradient = rawDataHighFidelity(i,dimHiFi+1+j) - rawDataHighFidelity(indexLowFi,dimLoFi+1+j);

				rawDataError(i,dimHiFi+1+j) = errorGradient;
			}

		}
	}

	rawDataError.save(inputFileNameError, csv_ascii);

	ifErrorDataIsSet = true;

	printMsg("Preparing error data is done...");
}


void MultiLevelModel::setGradientsOnLowFi(void){

	ifLowFidelityDataHasGradients = true;

}
void MultiLevelModel::setGradientsOnHiFi(void){

	ifHighFidelityDataHasGradients = true;
}
void MultiLevelModel::setGradientsOffLowFi(void){

	ifLowFidelityDataHasGradients = false;

}
void MultiLevelModel::setGradientsOffHiFi(void){

	ifHighFidelityDataHasGradients = false;

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


	if(dimHiFi != dimLoFi){

		std::cout<<"ERROR: dimHiFi != dimLoFi!\n";
		std::cout<<"dimHiFi = "<<dimHiFi<<"\n";
		std::cout<<"dimLoFi = "<<dimLoFi<<"\n";
		abort();

	}

	dim = dimHiFi;


}

void MultiLevelModel::readData(void){

	assert(ifBoundsAreSet);

	printMsg("Reading data for the multilevel model...");
	readHighFidelityData();
	readLowFidelityData();

	setDimensionsHiFiandLowFiModels();

	XLowFidelity =  rawDataLowFidelity.submat(0,0,NLoFi -1, dimLoFi -1);
	XHighFidelity = rawDataHighFidelity.submat(0,0,NHiFi -1, dimHiFi -1);

	XLowFidelity  = normalizeMatrix(XLowFidelity, xmin, xmax);
	XLowFidelity = (1.0/dim)*XLowFidelity;
	XHighFidelity = normalizeMatrix(XHighFidelity, xmin, xmax);
	XHighFidelity = (1.0/dim)*XHighFidelity;

	prepareErrorData();

	ifDataIsRead = true;


}

void MultiLevelModel::bindLowFidelityModel(void){


	if(ifLowFidelityDataHasGradients){

		lowFidelityModel= &surrogateModelAggregationLowFi;
		lowFidelityModel->setNameOfHyperParametersFile(label + "lowFi_aggregation_model_hyperparameters.csv");
		printMsg("Binding the low fidelity model with the aggregation model");

	}

	else{

		lowFidelityModel = &surrogateModelKrigingLowFi;
		lowFidelityModel->setNameOfHyperParametersFile(label + "lowFi_Kriging_model_hyperparameters.csv") ;

		printMsg("Binding the low fidelity model with the Kriging model");

	}



	ifLowFidelityModelIsSet = true;

}

void MultiLevelModel::bindErrorModel(void){



	if(ifLowFidelityDataHasGradients && ifHighFidelityDataHasGradients){

		errorModel = &surrogateModelAggregationError;

		printMsg("Binding the error model with the aggregation model");

	}

	else{

		errorModel = &surrogateModelKrigingError;

		printMsg("Binding the error model with the Kriging model");

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



void MultiLevelModel::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{

	std::cout<<"ERROR: MultiLevelModel::calculateExpectedImprovement is not implemented yet!\n";
	abort();


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

void MultiLevelModel::addNewSampleToData(rowvec newsample){




}


