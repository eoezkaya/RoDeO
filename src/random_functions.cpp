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

#include <cassert>

#include "random_functions.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

int generateRandomInt(int a, int b){
	assert(b>a);
	return(rand() % (b-a)+a);

}


int generateRandomInt(uvec indices){
	unsigned int size = indices.size();
	assert(size>0);
	int randomIndex = generateRandomInt(0, size);

	return(indices(randomIndex));

}



double generateRandomDouble(double a, double b) {

	double random = ((double) rand()) / (double) RAND_MAX;
	double diff = b - a;
	double r = random * diff;
	return a + r;
}


void generateRandomDoubleArray(double *xp,double a, double b, unsigned int dim) {

	for(unsigned int i=0; i<dim; i++) {

		xp[i] = generateRandomDouble(a,b);

	}


}



rowvec generateRandomRowVector(vec lb, vec ub){
	unsigned int dim = lb.size();
	rowvec x(dim);
	for(unsigned int i=0; i<dim; i++) {
		assert(lb(i) <= ub(i));
		x(i) = generateRandomDouble(lb(i), ub(i));
	}
	return x;

}

rowvec generateRandomRowVector(double lb, double ub, unsigned int dim){

	assert(lb <= ub);
	rowvec x(dim,fill::zeros);

	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb, ub);
	}
	return x;

}

vec generateRandomVector(vec lb, vec ub){
	unsigned int dim = lb.size();
	vec x(dim);
	for(unsigned int i=0; i<dim; i++) {
		assert(lb(i) <= ub(i));
		x(i) = generateRandomDouble(lb(i), ub(i));
	}
	return x;

}

vec generateRandomVector(double lb, double ub, unsigned int dim){

	assert(lb <= ub);
	vec x(dim,fill::zeros);

	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb, ub);
	}
	return x;

}


void generateRandomVector(vec lb, vec ub, unsigned int dim, double *x){


	for(unsigned int i=0; i<dim; i++) {
		assert(lb(i) < ub(i));
		x[i] = generateRandomDouble(lb(i), ub(i));
	}

}



/** generate a random number between xs and xe using the normal distribution
 *
 */
double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor){

	double sigma=fabs((xe-xs))/sigma_factor;
	double mu=(xe+xs)/2.0;

	if (sigma == 0.0) sigma=1.0;

	/* construct a trivial random generator engine from a time-based seed */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution<double> distribution (mu,sigma);
	return distribution(generator);
}

void generateKRandomIntegers(uvec &numbers, unsigned int N, unsigned int k){

	unsigned int numbersGenerated = 0;

	numbers.fill(0);


	while (numbersGenerated != k){


		int r = rand()%N;
#if 0
		printf("random number = %d\n",r);
#endif
		if (is_in_the_list(r, numbers) == -1 ){

			numbers(numbersGenerated) = r;
			numbersGenerated++;
#if 0
			printf("numbers =\n");
			numbers.print();
#endif
		}

	}

}





/** generate a random weight matrix: all elements other than the diagonal are positive, sum of all rows is equal to one
 *
 */
mat generateRandomWeightMatrix(unsigned int dim){

	mat weights(dim,dim,fill::randn);

	weights = abs(weights);
	for(unsigned int i=0; i<dim; i++){

		weights(i,i) = 0.0;
	}
	colvec rowSum = sum(weights,1);

	for(unsigned int i=0; i<dim; i++){
		for(unsigned int j=0; j<dim; j++){

			weights(i,j) = weights(i,j)/rowSum(i);
		}

	}


	return weights;

}


