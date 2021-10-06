/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 *  file is part of RoDeO
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


ConstraintDefinition::ConstraintDefinition(void){

	value = 0.0;

}


ConstraintDefinition::ConstraintDefinition(std::string nameInput, std::string ineqType, double valueInput){

	assert(isNotEmpty(nameInput));

	name = nameInput;
	inequalityType = ineqType;
	value = valueInput;

}

ConstraintDefinition::ConstraintDefinition(std::string definition){

	assert(!definition.empty());
	size_t found  = definition.find(">");
	size_t found2 = definition.find("<");
	size_t place;
	std::string nameBuf,typeBuf, valueBuf;

	if (found!= std::string::npos){

		place = found;
	}
	else if (found2 != std::string::npos){

		place = found2;
	}
	else{

		std::cout<<"ERROR: Something is wrong with the constraint definition!\n";
		abort();
	}
	nameBuf.assign(definition,0,place);
	nameBuf = removeSpacesFromString(nameBuf);

	typeBuf.assign(definition,place,1);
	valueBuf.assign(definition,place+1,definition.length() - nameBuf.length() - typeBuf.length());
	valueBuf = removeSpacesFromString(valueBuf);


	name = nameBuf;
	inequalityType = typeBuf;
	value = stod(valueBuf);


}

void ConstraintDefinition::print(void) const{

	std::cout<<"\nConstraint definition = \n";
	std::cout<<"ID = "<<ID<<"\n";
	std::cout<<name<<" "<<inequalityType<<" "<<value<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";
	std::cout<< "Output filename = "<<outputFilename<<"\n";
	std::cout<< "Executable name = "<<executableName<<"\n";

	if(!path.empty()){

		std::cout<< "Executable path = "<<path<<"\n";

	}

}



ConstraintFunction::ConstraintFunction(std::string name, unsigned int dimension)
: ObjectiveFunction(name, dimension){



}


void ConstraintFunction::readEvaluateOutput(Design &d) {

#if 0
	cout<<"Reading constraint function\n";
#endif

	assert(d.dimension == dim);


	if(ifAdjointMarkerIsSet == true){


		if(ifGradientAvailable == false && readMarkerAdjoint != "None"){

			cout << "ERROR: Adjoint marker is set for the constraint function but gradient is not available!\n";
			cout << "Did you set GRADIENT_AVAILABLE properly?\n";
			abort();

		}

	}


	if(ifGradientAvailable && ifMarkerIsSet){

		if(ifAdjointMarkerIsSet== false){

			std::cout << "ERROR: Adjoint marker not is set for a constraint function!\n";
			std::cout << "Did you set CONSTRAINT_FUNCTION_GRADIENT_READ_MARKER properly?\n";
			abort();

		}



	}


	/* input from file without using markers */

	if(checkIfMarkersAreNotSet()){

		readOutputWithoutMarkers(d);

	}

	/* input from file using markers (the values are separated with ',')*/
	else{

		readOutputWithMarkers(d);

	}

#if 0
	d.print();
#endif

}

void ConstraintFunction::readOutputWithoutMarkers(Design &outputDesignBuffer) const{

	assert(isNotEmpty(fileNameInputRead));
	std::ifstream inputFileStream(fileNameInputRead, ios::in);

	if (!inputFileStream.is_open()) {

		cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}


	rowvec constraintGradient = zeros<rowvec>(dim);

	double functionValue;
	inputFileStream >> functionValue;


	outputDesignBuffer.constraintTrueValues(ID) = functionValue;


	if(ifGradientAvailable){


		for(unsigned int i=0; i<dim; i++){

			inputFileStream >> constraintGradient(i);

		}

		abortIfHasNan(constraintGradient);

	}


	outputDesignBuffer.constraintGradients.push_back(constraintGradient);
	inputFileStream.close();


}


void ConstraintFunction::readOutputWithMarkers(Design &outputDesignBuffer) const{

	assert(isNotEmpty(fileNameInputRead));
	std::ifstream inputFileStream(fileNameInputRead, ios::in);

	if (!inputFileStream.is_open()) {

		cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}


	rowvec constraintGradient = zeros<rowvec>(dim);


	bool markerFound = false;
	bool markerAdjointFound = false;

	for( std::string line; getline( inputFileStream, line ); ){ /* check each line */

		size_t foundMarkerPosition = isMarkerFound(readMarker, line);

		if (foundMarkerPosition != std::string::npos){

			outputDesignBuffer.constraintTrueValues(ID) = getMarkerValue(line, foundMarkerPosition );
			markerFound = true;

		}


		if(ifGradientAvailable){

			/* check for constraint gradient marker */
			size_t foundMarkerPosition = isMarkerFound(readMarkerAdjoint, line);

			if (foundMarkerPosition != std::string::npos){

				constraintGradient = getMarkerAdjointValues(line, foundMarkerPosition );
				markerAdjointFound  = true;

			}
		}

	}
#if 0
	printVector(constraintGradient,"constraintGradient");
#endif

	outputDesignBuffer.constraintGradients.push_back(constraintGradient);

	if(ifGradientAvailable && !markerAdjointFound){

		cout<<"ERROR: No values can be read for the constraint gradient!\n";
		abort();


	}

	if(markerFound == false){

		cout<<"ERROR: No values can be read for the constraint function!\n";
		abort();


	}

	inputFileStream.close();


}

void ConstraintFunction::print(void) const {

	std::cout << "\n#####################################################\n";
	std::cout << "Constraint function ID = "<<ID<<"\n";
	std::cout << "Name: " << name << endl;
	std::cout << "Dimension: " << dim << endl;
	std::cout << "Type of constraint: " << inequalityType << " " <<value<< endl;
	std::cout << "Executable name: " << executableName << "\n";
	std::cout << "Executable path: " << executablePath << "\n";
	std::cout << "Input file name: " << fileNameDesignVector << "\n";
	std::cout << "Output file name: " << fileNameInputRead << "\n";
	std::cout << "Read marker: " <<readMarker <<"\n";
	if(ifGradientAvailable){
		std::cout<<"Uses gradient vector: Yes\n";
		std::cout << "Read marker for gradient: " <<readMarkerAdjoint <<"\n";

	}
	else{
		std::cout<<"Uses gradient vector: No\n";

	}


#if 0
	surrogateModel.printSurrogateModel();
#endif
	std::cout << "#####################################################\n\n";
}


bool ConstraintFunction::isInequalityTypeValid(const std::string &inequalityType){

	if (inequalityType == ">") return true;
	if (inequalityType == "<") return true;

	return false;

}

void ConstraintFunction::setParametersByDefinition(ConstraintDefinition inequalityConstraint){


	assert(isNotEmpty(inequalityConstraint.inequalityType));
	assert(isNotEmpty(inequalityConstraint.name));
	assert(isInequalityTypeValid(inequalityConstraint.inequalityType));


	ID = inequalityConstraint.ID;

	executableName = inequalityConstraint.executableName;
	executablePath = inequalityConstraint.path;
	fileNameDesignVector = inequalityConstraint.designVectorFilename;
	fileNameInputRead = inequalityConstraint.outputFilename;
	inequalityType = inequalityConstraint.inequalityType;
	name =  inequalityConstraint.name;

	value = inequalityConstraint.value;




	if(isNotEmpty(inequalityConstraint.marker)){

		readMarker = inequalityConstraint.marker;
		ifMarkerIsSet = true;

	}

	if(isNotEmpty(inequalityConstraint.markerForGradient)){


		readMarkerAdjoint = inequalityConstraint.markerForGradient;
		ifAdjointMarkerIsSet = true;

	}



	ifInequalityConstraintSpecified  = true;

	ifDefinitionIsSet = true;


}



double ConstraintFunction::getValue(void) const{

	return value;


}

std::string ConstraintFunction::getInequalityType(void) const{

	return inequalityType;


}


bool ConstraintFunction::checkFeasibility(double valueIn) const{

	assert(ifInequalityConstraintSpecified);

	bool result = false;
	if (inequalityType == "<") {

		if (valueIn < value) {

			result = true;

		}

	}

	if (inequalityType == ">") {

		if (valueIn > value) {

			result = true;

		}

	}

	return result;
}


void ConstraintFunction::evaluate(Design &d) {

	double functionValue = 0.0;

	rowvec x = d.designParameters;
	if (ifFunctionPointerIsSet) {

		functionValue = objectiveFunPtr(x.memptr());


		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

	}

	else if (executableName != "None" && fileNameDesignVector != "None") {

		if (ifRunNecessary) {

			std::string runCommand = getExecutionCommand();
#if 0
			cout<<"calling a system command\n";
#endif

			system(runCommand.c_str());

		}

#if 0
		cout<<"No need to call a system command\n";
#endif



	} else {

		cout<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
		abort();
	}


}

void ConstraintFunction::addDesignToData(Design &d){

	if(ifGradientAvailable){

		rowvec newsample = d.constructSampleConstraintWithGradient(ID);

		surrogateModelGradient.addNewSampleToData(newsample);

	}
	else{

		rowvec newsample = d.constructSampleConstraint(ID);

		surrogateModel.addNewSampleToData(newsample);
	}


}


