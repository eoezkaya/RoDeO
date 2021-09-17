/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


WeightedL1Norm:: WeightedL1Norm(){


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


void WeightedL1Norm::setTrainingData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	trainingData = inputMatrix;

}
void WeightedL1Norm:: setValidationData(mat inputMatrix){

	assert(inputMatrix.n_rows>0);
	validationData = inputMatrix;

}


void WeightedL1Norm::setNumberOfTrainingIterations(unsigned int value){

	nTrainingIterations = value;

}


unsigned int  WeightedL1Norm::getDimension(void) const{

	return dim;
}

vec WeightedL1Norm::getWeights(void) const{

	return weights;
}

void WeightedL1Norm::setWeights(vec wIn){

	assert(wIn.size() == dim);
	weights = wIn;
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

	assert(trainingData.n_rows>0);
	assert(validationData.n_rows>0);
	assert(validationData.n_cols == trainingData.n_cols);

	vec optimalWeights;
	double minError = LARGE;


	for(unsigned int i=0; i<nTrainingIterations; i++){

		generateRandomWeights();
		double error = calculateMeanSquaredErrorOnData();


		if(error < minError){

			if(ifDisplay){

				printVector(weights,"weights");
				std::cout<<"minError = "<<minError<<"\n";

			}

			optimalWeights = weights;
			minError = error;

		}


	}



	weights = optimalWeights;

	if(ifDisplay){

		printVector(weights,"optimal weights");

	}

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


double calculateMetricAdjoint(rowvec xi, rowvec xj, mat M, mat &Mb, double calculateMetricb) {

	int dim = xi.size();
	rowvec diff(dim);


	rowvec tempb(dim, fill::zeros);
	double calculateMetric;

	diff = xi-xj;
	colvec diffT= trans(diff);

	calculateMetric = dot(diff,M*diffT);
	double sumb = 0.0;
	sumb = calculateMetricb;
	tempb = sumb*diff;

	for (int i = dim-1; i > -1; --i) {
		double sumb = 0.0;
		sumb = tempb[i];
		tempb[i] = 0.0;
		for (int j = dim-1; j > -1; --j){

			Mb(i,j) += diff[j]*sumb;
		}
	}

	return calculateMetric;
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




