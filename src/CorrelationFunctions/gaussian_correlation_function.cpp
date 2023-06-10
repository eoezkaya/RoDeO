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

#include "gaussian_correlation_function.hpp"
#include <cassert>
#include "LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


void GaussianCorrelationFunction::initialize(void){

	assert(dim>0);
	vec thetaInit(dim); thetaInit.fill(1.0);
	theta = thetaInit;

}

void GaussianCorrelationFunction::setHyperParameters(vec input){

	assert(input.empty() == false);
	assert(input.size() == dim);

	theta = input ;
}

vec GaussianCorrelationFunction::getHyperParameters(void) const{

	return theta;
}


double GaussianCorrelationFunction::computeCorrelation(const rowvec &xi, const rowvec &xj) const {


	double sum = 0.0;
	for (unsigned int k = 0; k < dim; k++) {
		double r = (xi(k) - xj(k))*(xi(k) - xj(k));
		double prod = theta(k)*r;
		sum += prod;
	}
	return exp(-sum);
}



double GaussianCorrelationFunction::computeCorrelationDot(const rowvec &xi, const rowvec &xj, const rowvec &diffDirection) const {

	double sumd = 0.0;
	double sum  = 0.0;

	for (unsigned int k = 0; k < dim; k++) {

		sumd += -2.0*theta(k) * (xi(k) - xj(k))*diffDirection(k);
		sum  += theta(k) * pow(fabs(xi(k) - xj(k)), 2.0);
	}

	double correlation = exp(-sum);

	double derivative = -1.0*sumd*correlation;

	assert(!isinf(derivative));


	return -1.0*exp(-sum)*sumd;

}




double GaussianCorrelationFunction::computeCorrelationDotDot(const rowvec &xi, const rowvec &xj, const rowvec &firstDiffDirection, const rowvec &secondDiffDirection) const{

	double td = 0.0;
	double t = 0.0;
	double td0 = 0.0;
	double tdd = 0.0;
	double temp;
	for (unsigned int i = 0; i < dim; i++) {
		temp = 2.0*theta(i)*firstDiffDirection(i);
		tdd = tdd + temp*secondDiffDirection(i);
		td = td - temp*(xi(i)-xj(i));
		td0 = td0 - theta(i)*2.0*(xi(i)-xj(i))*secondDiffDirection(i);
		t += theta(i)*(xi(i)-xj(i))*(xi(i)-xj(i));

	}
	temp = exp(-t);
	double resultd = -(temp*td);
	double resultdd = -(temp*tdd-td*exp(-t)*td0);

	assert(!isinf(resultdd));

	return resultdd;

}

void GaussianCorrelationFunction::print(void) const{

	std::cout<<"Theta = \n";
	theta.print();

}
