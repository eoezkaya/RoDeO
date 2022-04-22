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

#include "linear_solver.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<cassert>
#include<iostream>

using namespace arma;

CholeskySystem::CholeskySystem(unsigned int dim){

	dimension = dim;

	A = zeros<mat>(dim,dim);
	L = zeros<mat>(dim,dim);


}


void CholeskySystem::setDimension(unsigned int dim){

	dimension = dim;

	A = zeros<mat>(dim,dim);
	L = zeros<mat>(dim,dim);

}

unsigned int CholeskySystem::getDimension(void) const{

	return dimension ;

}


bool CholeskySystem::checkDimension(unsigned int dim){

	if(dimension == dim) return true;
	else return false;

}

mat CholeskySystem::getLowerDiagonalMatrix(void) const{

	return L;

}

void CholeskySystem::setMatrix(mat input){

	assert(input.n_rows!=0);
	assert(input.n_cols!=0);
	assert(input.n_rows == input.n_cols);
	A = input;

	dimension = input.n_rows;
	L = zeros<mat>(dimension,dimension);

	ifMatrixIsSet = true;
}

mat CholeskySystem::getMatrix(void) const{

	return(A);
}



void CholeskySystem::factorize(void){

	assert(ifMatrixIsSet);

	bool cholesky_return  = chol(L, A, "lower");



	if (cholesky_return == false) {
		std::cout<<"ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed!\n";
		abort();
	}

	U = trans(L);

	ifFactorizationIsDone = true;


}

vec CholeskySystem::forwardSubstitution(vec rhs) const{


	vec x(dimension);
	x.fill(0.0);

	for (unsigned int i=0; i<dimension; i++){

		double sum = 0.0;
		for (unsigned int j=0; j<i; j++){

			sum += x(j)*L(i,j);

		}

		x(i) = (rhs(i) - sum)/L(i,i);


	}



	return x;
}

vec CholeskySystem::backwardSubstitution(vec rhs) const{

	vec x(dimension);
	x.fill(0.0);

	for (int i=dimension-1; i>=0; i--){

		double sum = 0.0;
		for (unsigned int j=i+1; j<dimension; j++){

			sum += x(j)*U(i,j);

		}

		x(i) = (rhs(i) - sum)/U(i,i);

	}

	return x;
}

vec CholeskySystem::solveLinearSystem(vec rhs) const{

	assert(ifFactorizationIsDone);

	vec v(dimension);
	vec x(dimension);

	/* solve L v = rhs */
	v = forwardSubstitution(rhs);
	/* solve U x = v */
	x = backwardSubstitution(v);

	return x;
}
