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

#include "./INCLUDE/linear_solver.hpp"
#include<cassert>
#include<iostream>
#include <cmath>
namespace Rodop {

CholeskySystem::CholeskySystem(unsigned dim){
	setDimension(dim);
}


void CholeskySystem::setDimension(unsigned dim){

	dimension = dim;

	A.reset();
	factorizationMatrix.reset();

	A.resize(dim, dim);
	factorizationMatrix.resize(dim, dim);

}

unsigned int CholeskySystem::getDimension(void) const{
	return dimension ;
}


bool CholeskySystem::checkDimension(unsigned int dim){

	if(dimension == dim) return true;
	else return false;

}

mat CholeskySystem::getFactorizedMatrix(void) const{
	return factorizationMatrix;
}

double CholeskySystem::calculateDeterminant(void){

	assert(ifFactorizationIsDone);
	double determinant = 0.0;

	vec diagonal = factorizationMatrix.diag();
	determinant = diagonal.product();

	return determinant*determinant;

}

double CholeskySystem::calculateLogDeterminant(void){

	assert(ifFactorizationIsDone);

	double determinant = 0.0;

	vec diagonal = factorizationMatrix.diag();

	for(unsigned int i=0; i<dimension; i++) {

		determinant+= log(diagonal(i));
	}

	return 2.0*determinant;

}



void CholeskySystem::setMatrix(mat input){

	A = input;

	dimension = input.getNRows();

	factorizationMatrix.reset();
	factorizationMatrix.resize(dimension,dimension);

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

	int ret;
	factorizationMatrix = A.cholesky(ret);
	if (ret == -1) {

		if(ifDisplay){
			std::cout<< "CholeskySystem: Factorization failed.\n";
		}
		ifFactorizationIsDone = false;
	}
	else{
		if(ifDisplay){
			std::cout<< "CholeskySystem: Factorization is successful.\n";
		}
		ifFactorizationIsDone = true;
	}

}


vec CholeskySystem::solveLinearSystem(const vec &b) const{

	assert(ifFactorizationIsDone);

	return factorizationMatrix.solveCholesky(b);
}




LUSystem::LUSystem(unsigned int dim){

	if (dim <= 0) {
		throw std::invalid_argument("LUSystem:: dimension must be positive.");
	}
	setDimension(dim);
}


void LUSystem::setDimension(unsigned int dim){

	if(dim<=0){
		throw std::invalid_argument("LUSystem:: dimension must be positive.");
	}

	dimension = dim;

	A.reset();
	factorizationMatrix.reset();

	A.resize(dim, dim);
	factorizationMatrix.resize(dim, dim);

	pivots.resize(dimension);

}

int LUSystem::getDimension(void) const{

	return dimension ;

}


bool LUSystem::checkDimension(unsigned int dim){

	if(dimension == dim) return true;
	else return false;

}

mat LUSystem::getFactorizedMatrix(void) const{
	return factorizationMatrix;
}



void LUSystem::setMatrix(mat input){

	A = input;

	dimension = input.getNRows();

	factorizationMatrix.reset();
	factorizationMatrix.resize(dimension,dimension);

	pivots.resize(dimension);

	ifMatrixIsSet = true;
	ifFactorizationIsDone = false;
}

mat LUSystem::getMatrix(void) const{

	return(A);
}

bool LUSystem::isFactorizationDone(void){

	return ifFactorizationIsDone;

}

void LUSystem::factorize(void){

	int return_value;
	factorizationMatrix = A.lu(pivots.data(), return_value);

	if(return_value == -1){

		ifFactorizationIsDone = false;
	}
	else{

		ifFactorizationIsDone = true;
	}

}


vec LUSystem::solveLinearSystem(const vec &b){

	if(b.getSize() != dimension){

		throw std::invalid_argument("LUSystem::solveLinearSystem: dimension of the rhs does not match.");
	}

	assert(ifFactorizationIsDone);

	return factorizationMatrix.solveLU(pivots.data(),b);

}


}

