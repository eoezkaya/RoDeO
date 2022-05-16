/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "metric.hpp"
#include "auxiliary_functions.hpp"
#include "bounds.hpp"





double WeightedL1NormOptimizer::calculateObjectiveFunctionInternal(vec input){

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


	nTrainingIterations = dim * 10000;
	if (nTrainingIterations > 500000) {
		nTrainingIterations = 500000;
	}


}

WeightedL1Norm::WeightedL1Norm(unsigned int d){

	dim = d;
	weights = zeros<vec>(dim);


}

WeightedL1Norm::WeightedL1Norm(vec w){

	assert(w.size()>0);
	dim = w.size();
	weights = w;


}

void WeightedL1Norm::initialize(unsigned int dimension){

	assert(dimension > 0);
	dim = dimension;
	weights = zeros<vec>(dim);
	weights.fill(1.0/dim);


}

void WeightedL1Norm::setDimensionIfNotSet() {
	if (dim == 0) {
		dim = trainingData.n_cols - 1;
	} else {
		assert(dim == trainingData.n_cols - 1);
	}
}

void WeightedL1Norm::setTrainingData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	trainingData = inputMatrix;

	setDimensionIfNotSet();

	ifTrainingDataIsSet = true;

}
void WeightedL1Norm:: setValidationData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	validationData = inputMatrix;

	setDimensionIfNotSet();
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
	this->ifNumberOfTrainingIterationsIsSet = true;
}


unsigned int  WeightedL1Norm::getDimension(void) const{

	return dim;
}

vec WeightedL1Norm::getWeights(void) const{

	return weights;
}

void WeightedL1Norm::setWeights(vec weightInput){

	assert(weightInput.size() == dim);
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
	for(unsigned int i=0; i<dim; i++){

		weights(i) = generateRandomDouble(0.0, 1.0);
		sumWeights+= weights(i);

	}

	for(unsigned int i=0; i<dim; i++){

		weights(i) = (weights(i))/sumWeights;

	}


}


double WeightedL1Norm::interpolateByNearestNeighbour(rowvec x) const{


	unsigned int N = trainingData.n_rows;
	mat X = trainingData.submat(0,0,N-1,dim-1);


	double minDist = LARGE;
	int indx = -1;

	for(unsigned int i=0; i<N; i++){

		rowvec xp = X.row(i);
		rowvec diff = x-xp;

		double dist = calculateNorm(diff);

		if(dist<minDist){

			indx = i;
			minDist = dist;
		}

	}

	return trainingData(indx,dim);


}

double WeightedL1Norm::calculateMeanL1ErrorOnData(void) const{

	assert(dim == validationData.n_cols-1);

	unsigned int numberOfValidationSamples = validationData.n_rows;

	mat X = validationData.submat(0,0,numberOfValidationSamples-1,dim-1);
	vec y = validationData.col(dim);

	double L1Error = 0.0;

	for(unsigned int i=0; i<numberOfValidationSamples; i++){

		rowvec xp = X.row(i);
		double fTilde = interpolateByNearestNeighbour(xp);
		L1Error += fabs(fTilde - y(i));

#if 0
		std::cout<<"fTilde = "<<fTilde<<" fExact = "<<y(i)<<" SE = "<<squaredError<<"\n";
#endif
	}


	return L1Error/numberOfValidationSamples ;

}

double WeightedL1Norm::calculateMeanSquaredErrorOnData(void) const{

	assert(dim == validationData.n_cols-1);

	unsigned int numberOfValidationSamples = validationData.n_rows;

	mat X = validationData.submat(0,0,numberOfValidationSamples-1,dim-1);
	vec y = validationData.col(dim);

	double squaredError = 0.0;

	for(unsigned int i=0; i<numberOfValidationSamples; i++){

		rowvec xp = X.row(i);
		double fTilde = interpolateByNearestNeighbour(xp);
		squaredError += (fTilde - y(i))*(fTilde - y(i));

#if 0
		std::cout<<"fTilde = "<<fTilde<<" fExact = "<<y(i)<<" SE = "<<squaredError<<"\n";
#endif
	}


	return squaredError/numberOfValidationSamples ;

}


void WeightedL1Norm::findOptimalWeights(void){


	assert(dim>0);

	WeightedL1NormOptimizer optimizerForWeights;

	Bounds boxConstraints(dim);
	boxConstraints.setBounds(0.0,1.0);
	optimizerForWeights.setBounds(boxConstraints);
	optimizerForWeights.setDimension(dim);

	unsigned int initialPopulationSize = dim*1000;
	optimizerForWeights.setInitialPopulationSize(initialPopulationSize);

	if(!ifNumberOfTrainingIterationsIsSet){

		initializeNumberOfTrainingIterations();
	}


	optimizerForWeights.setMaximumNumberOfGeneratedIndividuals(nTrainingIterations);

	optimizerForWeights.setMutationProbability(0.1);
	optimizerForWeights.setNumberOfGenerations(100);
	optimizerForWeights.setNumberOfNewIndividualsInAGeneration(dim*20);
	optimizerForWeights.setNumberOfDeathsInAGeneration(dim*20);

	optimizerForWeights.initializeWeightedL1NormObject(*this);
	optimizerForWeights.optimize();

	vec optimalWeights = optimizerForWeights.getBestDesignvector();


	double sumWeights = sum(optimalWeights);
	optimalWeights = optimalWeights/sumWeights;

	setWeights(optimalWeights);

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




