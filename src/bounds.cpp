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

#include "bounds.hpp"
#include "matrix_vector_operations.hpp"
#include<cassert>

Bounds::Bounds(){

}

Bounds::Bounds(unsigned int dim){

	dimension = dim;

}

Bounds::Bounds(vec lb, vec ub){

	assert( lb.size() == ub.size() );

	lowerBounds = lb;
	upperBounds = ub;
	dimension = lb.size();

	assert(checkIfBoundsAreValid());

	ifBoundsAreSet = true;


}


unsigned int Bounds::getDimension(void) const{

	return dimension;

}


void Bounds::setBounds(vec lowerBoundsInput, vec upperBoundsInput) {

	lowerBounds = lowerBoundsInput;
	upperBounds = upperBoundsInput;

	assert(lowerBounds.size() == upperBounds.size());

	dimension = lowerBounds.size();
	ifBoundsAreSet = true;

}

void Bounds::setBounds(double lowerBound, double upperBound) {

	assert(lowerBound < upperBound);
	assert(dimension > 0);

	lowerBounds =zeros<vec>(dimension);
	lowerBounds.fill(lowerBound);

	upperBounds =zeros<vec>(dimension);
	upperBounds.fill(upperBound);
	ifBoundsAreSet = true;

}

bool Bounds::areBoundsSet(void) const{

	return ifBoundsAreSet;

}




vec Bounds::getLowerBounds(void) const{

	return lowerBounds;


}

double Bounds::getLowerBound(unsigned int index) const{

	return lowerBounds(index);

}



vec Bounds::getUpperBounds(void) const{

	return upperBounds;

}

double Bounds::getUpperBound(unsigned int index) const{

	return upperBounds(index);

}

bool Bounds::checkIfBoundsAreValid(void) const{

	assert(dimension>0);

	for(unsigned int i=0; i<dimension; i++){


		if(lowerBounds(i) >= upperBounds(i) ) {

			return false;
		}

	}

	return true;

}

bool Bounds::isPointWithinBounds(const vec &inputVector) const{

	assert(inputVector.size() == dimension);

	for(unsigned int i=0; i<dimension; i++){

		if(inputVector(i) < lowerBounds(i)) return false;
		if(inputVector(i) > upperBounds(i)) return false;
	}

	return true;

}

void Bounds::print(void) const{

	printVector(this->lowerBounds,"Lower bounds");
	printVector(this->upperBounds,"Upper bounds");

}
