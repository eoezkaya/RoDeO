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
#include "auxiliary_functions.hpp"
#include "linear_solver.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<cassert>
#include<iostream>

using namespace arma;

CholeskySystem::CholeskySystem(unsigned int dim){

	dimension = dim;

	A.reset();
	L.reset();

	A = zeros<mat>(dim,dim);
	L = zeros<mat>(dim,dim);


}


void CholeskySystem::setDimension(unsigned int dim){

	dimension = dim;

	A.reset();
	L.reset();

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

mat CholeskySystem::getUpperDiagonalMatrix(void) const{

	return U;

}

double CholeskySystem::calculateDeterminant(void){

	assert(ifFactorizationIsDone);
	double determinant = 0.0;


	vec U_diagonal = U.diag();

	determinant = prod(U_diagonal);


	return determinant*determinant;

}

double CholeskySystem::calculateLogDeterminant(void){

	assert(ifFactorizationIsDone);

	double determinant = 0.0;

	vec U_diagonal = U.diag();

	for(unsigned int i=0; i<dimension; i++) {

		determinant+= log(U_diagonal(i));
	}

	return 2.0*determinant;

}



void CholeskySystem::setMatrix(mat input){

	assert(input.n_rows!=0);
	assert(input.n_cols!=0);
	assert(input.n_rows == input.n_cols);
	A = input;

	dimension = input.n_rows;
	L.reset();
	L = zeros<mat>(dimension,dimension);

	ifMatrixIsSet = true;
	ifFactorizationIsDone = false;
}

mat CholeskySystem::getMatrix(void) const{

	return(A);
}

bool CholeskySystem::isFactorizationDone(void){

	return ifFactorizationIsDone;

}

void CholeskySystem::factorize(void){

	assert(ifMatrixIsSet);

	bool cholesky_return  = chol(L, A, "lower");



	if (cholesky_return == false) {

		ifFactorizationIsDone = false;
	}
	else{

		U = trans(L);

		ifFactorizationIsDone = true;
	}




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




void SVDSystem::setMatrix(mat input){

	assert(input.n_rows!=0);
	assert(input.n_cols!=0);
	numberOfCols = input.n_cols;
	numberOfRows = input.n_rows;

	A = input;
	ifMatrixIsSet = true;
}

void SVDSystem::factorize(void){

	assert(ifMatrixIsSet);

	svd(U,sigma,V,A);

	ifFactorizationIsDone = true;

}

vec SVDSystem::solveLinearSystem(vec &b){

	setRhs(b);

	assert(ifFactorizationIsDone);

	vec solution(numberOfCols, fill::zeros);

	vec sigmaCut(sigma.size());
//	sigma.print("sigma");


	int numberOfActiveSingularValues = 0;
	for(unsigned int i=0; i<sigma.size();i++){

		if(sigma(i) < thresholdForSingularValues){

			sigmaCut(i) = 0.0;

		}
		else{

			sigmaCut(i) = 1.0/sigma(i);
			numberOfActiveSingularValues++;
		}

	}

//	sigmaCut.print("sigmaCut");


	for(unsigned int i=0; i<numberOfActiveSingularValues; i++){

		solution += dot(U.col(i),rhs)*sigmaCut(i)*V.col(i);

	}

	return solution;
}


vec SVDSystem::solveLinearSystem(void){

	assert(ifFactorizationIsDone);
	assert(ifRightHandSideIsSet);

	vec solution(numberOfCols, fill::zeros);

	vec sigmaCut(sigma.size());
//	sigma.print("sigma");


	int numberOfActiveSingularValues = 0;
	for(unsigned int i=0; i<sigma.size();i++){

		if(sigma(i) < thresholdForSingularValues){

			sigmaCut(i) = 0.0;

		}
		else{

			sigmaCut(i) = 1.0/sigma(i);
			numberOfActiveSingularValues++;
		}

	}

//	sigmaCut.print("sigmaCut");


	for(unsigned int i=0; i<numberOfActiveSingularValues; i++){

		solution += dot(U.col(i),rhs)*sigmaCut(i)*V.col(i);

	}

	return solution;
}


void SVDSystem::setRhs(vec input){
	rhs = input;
	ifRightHandSideIsSet = true;
}




void SVDSystem::setThresholdForSingularValues(double value) {

	assert(value<1.0);
	assert(value>0.0);
	thresholdForSingularValues = value;


}


double SVDSystem::calculateLogAbsDeterminant(void) const{

	assert(ifFactorizationIsDone);
	double Absdeterminant = 1.0;

	for(unsigned int i=0; i<sigma.size();i++){


		Absdeterminant = Absdeterminant*sigma(i);

		}


	return log(Absdeterminant);


}

