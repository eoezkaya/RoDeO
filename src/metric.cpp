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

#include "metric.hpp"
#include "auxiliary_functions.hpp"
#include "bounds.hpp"





double WeightedL1NormOptimizer::calculateObjectiveFunctionInternal(vec& input){

	weightedL1NormForCalculations.setWeights(input);

	double meanL1Error = weightedL1NormForCalculations.calculateMeanL1ErrorOnData();

	return meanL1Error;

}
void WeightedL1NormOptimizer::initializeWeightedL1NormObject(WeightedL1Norm input){

	assert(input.getDimension() > 0);
	assert(input.isTrainingDataSet());
	assert(input.isValidationDataSet());
	weightedL1NormForCalculations = input;
	ifWeightedL1NormForCalculationsIsSet = true;

}



bool WeightedL1NormOptimizer::isWeightedL1NormForCalculationsSet(void) const{

	return ifWeightedL1NormForCalculationsIsSet;

}


WeightedL1Norm:: WeightedL1Norm(){}

void WeightedL1Norm::initializeNumberOfTrainingIterations(void) {


	nTrainingIterations = dimension * 10000;
	if (nTrainingIterations > 500000) {
		nTrainingIterations = 500000;
	}


}






WeightedL1Norm::WeightedL1Norm(unsigned int d){

	dimension = d;
	weights = zeros<vec>(dimension);


}

WeightedL1Norm::WeightedL1Norm(vec w){

	assert(w.size()>0);
	dimension = w.size();
	weights = w;


}

void WeightedL1Norm::initialize(unsigned int dim){

	assert(dim > 0);
	dimension = dim;
	weights = zeros<vec>(dim);
	weights.fill(1.0/dim);


}


void WeightedL1Norm::setTrainingData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	assert(dimension == inputMatrix.n_cols - 1);
	trainingData = inputMatrix;
	inputTrainingData = trainingData.submat(0,0,trainingData.n_rows-1,dimension-1);
	outputTrainingData = trainingData.col(dimension);
	ifTrainingDataIsSet = true;

}
void WeightedL1Norm:: setValidationData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	assert(dimension == inputMatrix.n_cols - 1);
	validationData = inputMatrix;
	inputValidationData = validationData.submat(0,0,validationData.n_rows-1,dimension-1);
	outputValidationData = validationData.col(dimension);
	ifValidationDataIsSet = true;
}

bool WeightedL1Norm::isTrainingDataSet(void) const{

	return ifTrainingDataIsSet;


}

bool WeightedL1Norm::isValidationDataSet(void) const{

	return ifValidationDataIsSet;


}

bool WeightedL1Norm::isNumberOfTrainingIterationsSet(void) const{

	return ifNumberOfTrainingIterationsIsSet;

}



void WeightedL1Norm::setNumberOfTrainingIterations(unsigned int value){

	nTrainingIterations = value;
	ifNumberOfTrainingIterationsIsSet = true;
}


unsigned int  WeightedL1Norm::getDimension(void) const{

	return dimension;
}

vec WeightedL1Norm::getWeights(void) const{

	return weights;
}

void WeightedL1Norm::setDimension(unsigned int dim){

	dimension = dim;
	weights = zeros<vec>(dim);

}

void WeightedL1Norm::setNumberOfThreads(unsigned int n){

	numberOfThreads = n;


}

void WeightedL1Norm::setWeights(vec weightInput){

	assert(weightInput.size() == dimension);
	weights = weightInput;
}

double WeightedL1Norm::calculateNorm(const rowvec &x) const{

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += weights(i)*fabs(x(i));

	}

	return sum;
}

void WeightedL1Norm::generateRandomWeights(void){


	double sumWeights=0.0;
	for(unsigned int i=0; i<dimension; i++){

		weights(i) = generateRandomDouble(0.0, 1.0);
		sumWeights+= weights(i);

	}

	for(unsigned int i=0; i<dimension; i++){

		weights(i) = (weights(i))/sumWeights;

	}


}

int WeightedL1Norm::findNearestNeighbor(const rowvec &x) const {
	unsigned int N = trainingData.n_rows;
	double minDist = LARGE;
	int indx = -1;
	for (unsigned int i = 0; i < N; i++) {
		rowvec diff = x - inputTrainingData.row(i);
		double dist = calculateNorm(diff);
		if (dist < minDist) {
			indx = i;
			minDist = dist;
		}
	}
	return indx;
}

double WeightedL1Norm::interpolateByNearestNeighbor(rowvec x) const{

	assert(ifTrainingDataIsSet);

	int indx = findNearestNeighbor(x);
	return trainingData(indx,dimension);


}

double WeightedL1Norm::calculateMeanL1ErrorOnData(void) const{

	assert(ifValidationDataIsSet);
	assert(dimension == validationData.n_cols-1);

	unsigned int numberOfValidationSamples = validationData.n_rows;

	double L1Error = 0.0;

	for(unsigned int i=0; i<numberOfValidationSamples; i++){

		double fTilde = interpolateByNearestNeighbor(inputValidationData.row(i));
		L1Error += fabs(fTilde - outputValidationData(i));

	}

	return L1Error/numberOfValidationSamples ;

}

double WeightedL1Norm::calculateMeanSquaredErrorOnData(void) const{

	assert(dimension == validationData.n_cols-1);

	unsigned int numberOfValidationSamples = validationData.n_rows;

	mat X = validationData.submat(0,0,numberOfValidationSamples-1,dimension-1);
	vec y = validationData.col(dimension);

	double squaredError = 0.0;

	for(unsigned int i=0; i<numberOfValidationSamples; i++){

		rowvec xp = X.row(i);
		double fTilde = interpolateByNearestNeighbor(xp);
		squaredError += (fTilde - y(i))*(fTilde - y(i));

#if 0
		std::cout<<"fTilde = "<<fTilde<<" fExact = "<<y(i)<<" SE = "<<squaredError<<"\n";
#endif
	}


	return squaredError/numberOfValidationSamples ;

}




void WeightedL1Norm::findOptimalWeights(void){


	assert(dimension>0);

	omp_set_num_threads(numberOfThreads);

	double globalBestL1error = LARGE;
	vec globalOptimalWeights(dimension);

	unsigned int initialPopulationSize = dimension*1000;

	if(!ifNumberOfTrainingIterationsIsSet){

		initializeNumberOfTrainingIterations();
	}

#pragma omp parallel for
		for(unsigned int iThread = 0; iThread<numberOfThreads; iThread++){


			WeightedL1NormOptimizer optimizerForWeights;

			Bounds boxConstraints(dimension);
			boxConstraints.setBounds(0.0,1.0);
			optimizerForWeights.setBounds(boxConstraints);
			optimizerForWeights.setDimension(dimension);


			optimizerForWeights.setInitialPopulationSize(initialPopulationSize);
			optimizerForWeights.setMaximumNumberOfGeneratedIndividuals(nTrainingIterations);
			optimizerForWeights.setMutationProbability(0.1);
			optimizerForWeights.setNumberOfGenerations(10);
			optimizerForWeights.setNumberOfNewIndividualsInAGeneration(dimension*20);
			optimizerForWeights.setNumberOfDeathsInAGeneration(dimension*100);


			optimizerForWeights.initializeWeightedL1NormObject(*this);
			optimizerForWeights.optimize();

			EAIndividual solution = optimizerForWeights.getSolution();
			double bestL1Error = solution.getObjectiveFunctionValue();
			vec optimalWeights = solution.getGenes();

			double sumWeights = sum(optimalWeights);
			optimalWeights = optimalWeights/sumWeights;
#pragma omp critical
			{

				if(bestL1Error < globalBestL1error){
					globalBestL1error = bestL1Error;
					globalOptimalWeights = optimalWeights;

				}


			}
		}

	setWeights(globalOptimalWeights);

	omp_set_num_threads(1);

}





double calculateL1norm(const rowvec &x){

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += fabs(x(i));

	}

	return sum;
}

double calculateWeightedL1norm(const rowvec &x, vec w){

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += w(i)*fabs(x(i));

	}

	return sum;
}


double calculateMetric(rowvec &xi,rowvec &xj, mat M){

	rowvec diff= xi-xj;
	colvec diffT= trans(diff);

	return dot(diff,M*diffT);

}






unsigned int findNearestNeighborL1(const rowvec &xp, const mat &X){

	assert(X.n_rows>0);

	unsigned int index = -1;
	double minL1Distance = LARGE;



	for(unsigned int i=0; i<X.n_rows; i++){

		rowvec x = X.row(i);

		rowvec xdiff = xp -x;

		double L1distance = calculateL1norm(xdiff);
		if(L1distance< minL1Distance){

			minL1Distance = L1distance;
			index = i;

		}

	}


	return index;


}




