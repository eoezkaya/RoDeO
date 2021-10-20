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
#include "random_functions.hpp"
#include "matrix_vector_operations.hpp"
#include <cassert>

unsigned int StandardTestFunction::getDimension(void) const{

	return dimension;

}


Bounds StandardTestFunction::getBoxConstraints(void) const{

	return boxConstraints;
}


void StandardTestFunction::generateSamples(unsigned int howmanySamples) {

	assert(boxConstraints.areBoundsSet());

	unsigned int sizeOfASample = dimension+1;
	mat samplesMatrix(howmanySamples,sizeOfASample);

	for(unsigned int i=0; i<howmanySamples; i++){

		rowvec sampleGenerated(sizeOfASample, fill::zeros);

		rowvec x = generateRandomRowVector(boxConstraints.getLowerBounds(), boxConstraints.getUpperBounds());
		copyRowVector(sampleGenerated,x);
		sampleGenerated(dimension) = evaluate(x);
		samplesMatrix.row(i) = sampleGenerated;


	}


	samples = samplesMatrix;


}

void StandardTestFunction::generateSamplesWithGradient(unsigned int howmanySamples) {

	assert(boxConstraints.areBoundsSet());

	unsigned int sizeOfASample = 2*dimension+1;
	mat samplesMatrix(howmanySamples,sizeOfASample);

	for(unsigned int i=0; i<howmanySamples; i++){

		rowvec sampleGenerated(sizeOfASample, fill::zeros);

		rowvec x = generateRandomRowVector(boxConstraints.getLowerBounds(), boxConstraints.getUpperBounds());
		copyRowVector(sampleGenerated,x);
		sampleGenerated(dimension) = evaluate(x);
		rowvec grad = evaluateGradient(x);
		copyRowVector(sampleGenerated, grad, dimension+1);
		samplesMatrix.row(i) = sampleGenerated;

	}

	samples = samplesMatrix;

}



mat StandardTestFunction::getSamples(void) const{

	return samples;

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


rowvec StandardTestFunction::evaluateGradient(rowvec x) const{

	rowvec gradient(dimension);
	double epsilon;

	for(unsigned int i=0; i<dimension; i++){

		double xsave = x(i);
		epsilon = x(i)*0.001;
		x(i) += epsilon;
		double yplus  = evaluate(x);
		x(i) -= 2.0*epsilon;
		double yminus = evaluate(x);
		gradient(i) = (yplus- yminus)/(2.0*epsilon);


	}


	return gradient;

}


HimmelblauFunction::HimmelblauFunction(){

	dimension = 2;

}


double HimmelblauFunction::evaluate(rowvec x) const{

	assert(x.size() == dimension);
	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );


}

void HimmelblauFunction::evaluateGradient(rowvec x, rowvec gradient) const{

	assert(x.size() == gradient.size());

	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	gradient[0] = tempb0 + 2*x[0]*tempb;
	gradient[1] = 2*x[1]*tempb0 + tempb;


}


