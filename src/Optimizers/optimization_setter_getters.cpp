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


#include "./INCLUDE/optimization.hpp"

void Optimizer::setName(std::string problemName){
	name = problemName;
}

void Optimizer::setMaximumNumberOfIterations(unsigned int maxIterations){

	maxNumberOfSamples = maxIterations;

}


void Optimizer::setMaximumNumberOfIterationsLowFidelity(unsigned int maxIterations){

	maxNumberOfSamplesLowFidelity =  maxIterations;


}



void Optimizer::setMaximumNumberOfInnerIterations(unsigned int maxIterations){

	iterMaxAcquisitionFunction = maxIterations;

}


void Optimizer::setFileNameDesignVector(std::string filename){

	assert(!filename.empty());
	designVectorFileName = filename;

}


void Optimizer::setBoxConstraints(Bounds boxConstraints){

	lowerBounds = boxConstraints.getLowerBounds();
	upperBounds = boxConstraints.getUpperBounds();


	assert(ifObjectFunctionIsSpecied);
	objFun.setParameterBounds(boxConstraints);

	if(isConstrained()){

		for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

			it->setParameterBounds(boxConstraints);
		}

	}

	globalOptimalDesign.setBoxConstraints(boxConstraints);
	ifBoxConstraintsSet = true;
}


void Optimizer::setDisplayOn(void){
	output.ifScreenDisplay = true;
}
void Optimizer::setDisplayOff(void){
	output.ifScreenDisplay = false;
}


void Optimizer::setNumberOfThreads(unsigned int n){
	numberOfThreads = n;
}

void Optimizer::setDimension(unsigned int dim){

	dimension = dim;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	initializeBoundsForAcquisitionFunctionMaximization();
	iterMaxAcquisitionFunction = dimension*10000;

	globalOptimalDesign.setDimension(dim);
	currentBestDesign.setDimension(dim);

}
