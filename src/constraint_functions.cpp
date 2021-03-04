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

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "constraint_functions.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;



ConstraintFunction::ConstraintFunction(std::string name, unsigned int dimension)
: ObjectiveFunction(name, dimension){



}


void ConstraintFunction::readEvaluateOutput(Design &d) {


	assert(ID>0);
	assert(!fileNameInputRead.empty());
	assert(int(d.constraintTrueValues.size()) >= ID);

	unsigned int totalNumberOfEntriesToRead;
	if(ifGradientAvailable){

		totalNumberOfEntriesToRead = readOutputStartIndex + dim+1;
	}
	else{

		totalNumberOfEntriesToRead = readOutputStartIndex+1;

	}

	std::ifstream ifile(fileNameInputRead, std::ios::in);

	if (!ifile.is_open()) {

		std::cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}

	vec bufferRead(totalNumberOfEntriesToRead);

	for (unsigned int i = 0; i < totalNumberOfEntriesToRead ; i++) {

		ifile >> bufferRead(i);
	}
#if 0
	printVector(bufferRead);
#endif
	d.constraintTrueValues(ID-1) = bufferRead(readOutputStartIndex);


	if(ifGradientAvailable){

		rowvec constraintGradient(dim);

		for(unsigned int i=0; i<dim; i++){

			constraintGradient(i) = bufferRead(readOutputStartIndex+i+1);

		}

		if(constraintGradient.has_nan()){

			std::cout<<"ERROR: NaN in constraint gradient evaluation!\n";
			abort();

		}

		d.constraintGradients.push_back(constraintGradient);
	}
	else{

		rowvec constraintGradient(dim);
		constraintGradient.fill(0.0);
		d.constraintGradients.push_back(constraintGradient);
	}

}

void ConstraintFunction::print(void) const {

	std::cout << "#####################################################\n";
	std::cout << std::endl;
	std::cout << "Constraint Function\n";
	std::cout << "ID: " << ID << std::endl;
	std::cout << "Name: " << name << std::endl;
	std::cout << "Dimension: " << dim << std::endl;
	std::cout << "Type of constraint: " << inequalityType << " " << targetValue<< std::endl;
	std::cout << "Executable name: " << executableName << "\n";
	std::cout << "Executable path: " << executablePath << "\n";
	std::cout << "Input file name: " << fileNameDesignVector << "\n";
	std::cout << "Output file name: " << fileNameInputRead << "\n";
	if(ifGradientAvailable){
		std::cout<<"Uses gradient vector: Yes\n";

	}
	else{
		std::cout<<"Uses gradient vector: No\n";

	}

	std::cout << "Shares executable with:";
	for (std::vector<int>::const_iterator i =
			IDToFunctionsShareOutputExecutable.begin();
			i != IDToFunctionsShareOutputExecutable.end(); ++i)
		std::cout << " " << *i << ' ';

	std::cout<<"\n";
	std::cout<<"readOutputStartIndex = "<<readOutputStartIndex<<"\n";

#if 0
	surrogateModel.printSurrogateModel();
#endif
	std::cout << "#####################################################\n";
}


void ConstraintFunction::setInequalityConstraint(std::string inequalityStatement){

		inequalityStatement = removeSpacesFromString(inequalityStatement);

		char type = inequalityStatement.front();
		assert(type == '<' || type == '>');
		inequalityStatement.erase(0,1);
		inequalityType = type;
		double value = std::stod(inequalityStatement);
		this->targetValue = value;

		ifInequalityConstraintSpecified  = true;

	}


bool ConstraintFunction::checkFeasibility(double value) const{

	assert(ifInequalityConstraintSpecified);

	bool result = false;
	if (inequalityType == "<") {

		if (value < targetValue) {

			result = true;

		}

	}

	if (inequalityType == ">") {

		if (value > targetValue) {

			result = true;

		}

	}

	return result;
}



