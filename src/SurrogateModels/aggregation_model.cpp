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
#include <random>
#include <map>
using namespace arma;
using std::cout;


#include "aggregation_model.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "linear_regression.hpp"


void AggregationModel::setDimension(unsigned int dim){

	dimension = dim;
	data.setDimension(dim);
	krigingModel.setDimension(dim);

}


void AggregationModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());
	assert(boxConstraintsInput.getDimension() == dimension);

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);

	krigingModel.setBoxConstraints(boxConstraintsInput);

}


void AggregationModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	assert(dimension>0);

	data.setGradientsOn();
	data.readData(filenameDataInput);
	numberOfSamples = data.getNumberOfSamples();

	mat rawTrainingData = data.getRawData();
	mat trainingDataFortheKrigingModel = rawTrainingData.submat(0, 0, numberOfSamples-1, dimension);

	//	rawTrainingData.print("rawData");
	//	trainingDataFortheKrigingModel.print("rawDataKriging");

	trainingDataFortheKrigingModel.save(filenameTrainingForKrigingModel, csv_ascii);

	krigingModel.setNameOfInputFile(filenameTrainingForKrigingModel);
	krigingModel.readData();

	ifDataIsRead = true;

}

void AggregationModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalize();
	krigingModel.normalizeData();

	ifNormalized = true;
}





void AggregationModel::setNameOfInputFile(std::string filename){

	assert(isNotEmpty(filename));
	filenameDataInput  = filename;
}

void AggregationModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;
	krigingModel.setNumberOfTrainingIterations(nIters);

}

void AggregationModel::setNameOfHyperParametersFile(std::string label){

	assert(isNotEmpty(label));

	string filename = label + "_aggregation_hyperparameters.csv";

	filenameHyperparameters = filename;

	output.printMessage("Name of the hyperparameter file is set as: ", filename);


}


void AggregationModel::determineRhoBasedOnData(void){

	output.printMessage("Determining the hyper-parameter rho for the aggregation model...");

	unsigned int numberOfProbes = 1000;
	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();
	mat gradientData = data.getGradientMatrix();

	vec probeDistances(numberOfProbes);

	for(unsigned int i=0; i<numberOfProbes; i++){

		rowvec xp = generateRandomRowVector(0.0, 1.0/dim, dim);

		int indexNearestNeighbor = findNearestNeighbor(xp);

		rowvec xdiff = xp - data.getRowX(indexNearestNeighbor);

		double distance = weightedL1norm.calculateNorm(xdiff);

		probeDistances(i)= distance;

	}

	double averageDistance = mean(probeDistances);

	output.printMessage("averageDistance = ", averageDistance);



	/* obtain gradient statistics */
	double sumNormGrad = 0.0;
	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec gradVec= gradientData.row(i);


		double normGrad= calculateL1norm(gradVec);
		sumNormGrad += normGrad;

	}

	double averageNormGrad = sumNormGrad/numberOfSamples;
	output.printMessage("averageNormGrad = ", averageNormGrad);



	/* Note that
	 *
	 *  w2 = exp(-rho*norm(distance)*norm(grad))
	 *
	 *
	 */


	rho = -2.0*log(0.0001)/ (max(probeDistances) * averageNormGrad);

	output.printMessage("rho = ", rho);




}

void AggregationModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(dimension>0);
	assert(numberOfSamples>0);

	krigingModel.initializeSurrogateModel();

	numberOfHyperParameters = dimension + 1;
	weightedL1norm.initialize(dimension);

	ifInitialized = true;

#if 0
	printSurrogateModel();
#endif

}

void AggregationModel::prepareTrainingAndTestData(void){

	assert(ifDataIsRead);

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

	unsigned int numberOfSamplesInTestSet = 2*(N/5);
	unsigned int numberOfSamplesInTrainingSet = N - numberOfSamplesInTestSet;

	mat rawData = data.getRawData();


	mat shuffledrawData = shuffle(rawData);

	Bounds boxConstraints = data.getBoxConstraints();

	mat trainingRawData = shuffledrawData.submat( 0, 0, numberOfSamplesInTrainingSet-1, dim );
	mat testRawData = shuffledrawData.submat( numberOfSamplesInTrainingSet, 0, N-1, dim );

	mat trainingData = normalizeMatrix(trainingRawData, boxConstraints);

	mat testData     = normalizeMatrix(testRawData, boxConstraints);


	weightedL1norm.setTrainingData(trainingData);
	weightedL1norm.setValidationData(testData);

	output.printMessage("Training and test data for the L1 norm training are ready...");

}



void AggregationModel::printSurrogateModel(void) const{



}

void AggregationModel::printHyperParameters(void) const{

	output.printMessage("Hyperparameters of the aggregation model...");
	krigingModel.printHyperParameters();
	output.printMessage("rho", rho);

}
void AggregationModel::saveHyperParameters(void) const{

	unsigned int dim = data.getDimension();
	assert(numberOfHyperParameters == dim+1);

	vec  L1NormWeights = weightedL1norm.getWeights();

	vec saveBuffer(numberOfHyperParameters);
	saveBuffer(0) = rho;

	for(unsigned int i=1; i<dim+1; i++){

		saveBuffer(i) = L1NormWeights(i-1);

	}


	output.printMessage("Saving the hyperparameters of the aggregation model...");

	saveBuffer.save(filenameHyperparameters,csv_ascii);



}

void AggregationModel::loadHyperParameters(void){

	unsigned int dim = data.getDimension();
	assert(numberOfHyperParameters == dim+1);
	assert(isNotEmpty(filenameHyperparameters));

	vec  L1NormWeights(dim);

	vec loadBuffer(numberOfHyperParameters);

	if(file_exist(filenameHyperparameters.c_str())){

		loadBuffer.load(filenameHyperparameters,csv_ascii);

		if(loadBuffer.size() == numberOfHyperParameters){

			rho = loadBuffer(0);
			for(unsigned int i=1; i<dim+1; i++){

				L1NormWeights(i-1) = loadBuffer(i);

			}


			weightedL1norm.setWeights(L1NormWeights);

		}
		else{

			std::cout<<"WARNING: Number of entries in the hyperparameter file is inconsistent with the problem dimension!\n";

		}


	}

}

void AggregationModel::updateAuxilliaryFields(void){

	krigingModel.updateAuxilliaryFields();


}


vec AggregationModel::getL1NormWeights(void) const{

	return weightedL1norm.getWeights();


}


void AggregationModel::train(void){

	assert(ifDataIsRead);
	assert(ifNormalized);
	assert(ifInitialized);

	output.printMessage("Training the aggregation model: ", name);

	krigingModel.train();


	determineRhoBasedOnData();

	ifModelTrainingIsDone = true;

}


void AggregationModel::setRho(double value){

	rho = value;

}



double AggregationModel::calculateMinimumDistanceToNearestPoint(const rowvec &x, int index) const{

	rowvec xNearestPoint    = data.getRowX(index);

	rowvec xDiff = x - xNearestPoint;

	return weightedL1norm.calculateNorm(xDiff);

}


double AggregationModel::calculateDualModelEstimate(const rowvec &x, int index) const{

	vec y = data.getOutputVector();

	Bounds boxConstraints = data.getBoxConstraints();

	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	rowvec xp = normalizeRowVectorBack(x, xmin, xmax);


	double yNearestPoint = y(index);


	rowvec gradNearestPoint = data.getRowGradientRaw(index);
	rowvec xNearestPointRaw = data.getRowXRaw(index);

	xp.print("xp = ");
	xNearestPointRaw.print("x nearest = ");
	gradNearestPoint.print("grad nearest");
	printScalar(yNearestPoint);



	rowvec xDiffRaw = xp - xNearestPointRaw;

	return yNearestPoint+ dot(xDiffRaw,gradNearestPoint);

}

double AggregationModel::interpolate(rowvec x) const{

	int indexNearestPoint = findNearestNeighbor(x);

	double estimateKrigingModel = krigingModel.interpolate(x);
	double estimateDualModel = calculateDualModelEstimate(x,indexNearestPoint);

	printTwoScalars(estimateKrigingModel, estimateDualModel);

	double w2 = calculateDualModelWeight(x,indexNearestPoint);
	double w1 = 1.0 - w2;

	printTwoScalars(w1, w2);

	return w1*estimateKrigingModel + w2 * estimateDualModel;

}



void AggregationModel::interpolateWithVariance(rowvec x, double *estimateAggregationModel,double *ssqr) const{


	int indexNearestPoint = findNearestNeighbor(x);

	double estimateKrigingModel = 0.0;

	/* model uncertainty is computed using the Kriging model only */
	krigingModel.interpolateWithVariance(x,&estimateKrigingModel,ssqr);
	double estimateDualModel = calculateDualModelEstimate(x,indexNearestPoint);

	double w2 = calculateDualModelWeight(x,indexNearestPoint);
	double w1 = 1.0 - w2;


	*estimateAggregationModel = w1*estimateKrigingModel + w2 * estimateDualModel;



}

double AggregationModel::calculateDualModelWeight(const rowvec &x, int index) const{

	double minimumDistance = calculateMinimumDistanceToNearestPoint(x,index);
	rowvec gradNearestPoint = data.getRowGradient(index);
	double normGradient = weightedL1norm.calculateNorm(gradNearestPoint);

	return  exp(-rho*minimumDistance*normGradient);

}





unsigned int AggregationModel::findNearestNeighbor(const rowvec &xp) const{

	unsigned int index = 0;
	double minL1Distance = LARGE;

	unsigned int N = data.getNumberOfSamples();

	for(unsigned int i=0; i<N; i++){

		rowvec xdiff = xp - data.getRowX(i);

		double L1distance = weightedL1norm.calculateNorm(xdiff);

		if(L1distance < minL1Distance){

			minL1Distance = L1distance;
			index = i;

		}

	}

	return index;

}



void AggregationModel::addNewSampleToData(rowvec newSample){

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

	assert(newSample.size() == 2*dim+1);

	/* avoid points that are too close to each other */

	bool flagTooClose=false;
	for(unsigned int i=0; i<N; i++){

		rowvec sample = data.getRowRawData(i);


		if(checkifTooCLose(sample, newSample)) {

			flagTooClose = true;
		}

	}

	if(!flagTooClose){

		appendRowVectorToCSVData(newSample, filenameDataInput);

		updateModelWithNewData();

	}
	else{

		std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

	}

}

void AggregationModel::addNewLowFidelitySampleToData(rowvec newsample){


	assert(false);

}


void AggregationModel::updateModelWithNewData(void){

	readData();
	normalizeData();

	krigingModel.updateModelWithNewData();

}



