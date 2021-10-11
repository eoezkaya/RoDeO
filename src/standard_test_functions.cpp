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

#include "standard_test_functions.hpp"
#include "bounds.hpp"
#include <cassert>

unsigned int StandardTestFunction::getDimension(void) const{

	return dimension;

}


Bounds StandardTestFunction::getBoxConstraints(void) const{

	return boxConstraints;
}

void StandardTestFunction::setBoxConstraints(double lowerBound, double upperBound) {

	vec lb(dimension);
	vec ub(dimension);

	lb.fill(lowerBound);
	ub.fill(upperBound);

	Bounds boxConstraints(lb, ub);
	setBoxConstraints(boxConstraints);

}

void StandardTestFunction::setBoxConstraints(Bounds boxConstraintsInput) {

	assert(boxConstraintsInput.areBoundsSet());
	boxConstraints = boxConstraintsInput;


}



HimmelblauFunction::HimmelblauFunction(){

	dimension = 2;

}


double HimmelblauFunction::evaluate(rowvec x) const{

	assert(x.size() == dimension);
	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );


}
