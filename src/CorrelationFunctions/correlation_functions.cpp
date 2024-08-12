/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>

#include "correlation_functions.hpp"
#include "vector_operations.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


CorrelationFunctionBase::CorrelationFunctionBase(){}



bool CorrelationFunctionBase::isInputSampleMatrixSet(void) const{

	return ifInputSampleMatrixIsSet;

}


void CorrelationFunctionBase::setInputSampleMatrix(mat input){

	assert(input.empty() == false);
	X = input;
	N = X.n_rows;
	dim = X.n_cols;
	correlationMatrix.reset();
	correlationMatrix = zeros<mat>(N,N);
	ifInputSampleMatrixIsSet = true;


}

void CorrelationFunctionBase::setDimension(unsigned int input){

	dim = input;
}




void CorrelationFunctionBase::computeCorrelationMatrix(void){

	assert(isInputSampleMatrixSet());

	mat I = eye(N,N);

	correlationMatrix = I;

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = i + 1; j < N; j++) {

			double correlation = computeCorrelation(X.row(i), X.row(j));
			correlationMatrix (i, j) = correlation;
			correlationMatrix (j, i) = correlation;
		}

	}


	correlationMatrix  += I*epsilon;

}




vec CorrelationFunctionBase::computeCorrelationVector(const rowvec &xp) const{

	assert(isInputSampleMatrixSet());
	vec r(N);

	for(unsigned int i=0;i<N;i++){

		r(i) = computeCorrelation(xp, X.row(i) );

	}

	return r;


}



void CorrelationFunctionBase::setEpsilon(double value){

	assert(value>=0.0);
	assert(value<1.0);
	epsilon = value;

}


mat  CorrelationFunctionBase::getCorrelationMatrix(void) const{

	return correlationMatrix;

}








