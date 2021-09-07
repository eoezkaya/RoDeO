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

ConstraintDefinition::ConstraintDefinition(void){

	this->value = 0.0;

}


ConstraintDefinition::ConstraintDefinition(std::string name, std::string ineqType, double value){

	this->name = name;
	this->inequalityType = ineqType;
	this->value = value;

}

ConstraintDefinition::ConstraintDefinition(std::string definition){

	assert(!definition.empty());
	std::size_t found  = definition.find(">");
	std::size_t found2 = definition.find("<");
	std::size_t place;
	std::string nameBuf,typeBuf, valueBuf;

	if (found!=std::string::npos){

		place = found;
	}
	else if (found2!=std::string::npos){

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
	value = std::stod(valueBuf);


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
	std::cout<<"Reading constraint function\n";
#endif

	assert(!fileNameInputRead.empty());
	assert(d.dimension == dim);
	assert(objectiveFunPtr == empty);

	rowvec constraintGradient(dim);
	constraintGradient.fill(0.0);


	if(ifAdjointMarkerIsSet == true){


		if(ifGradientAvailable == false && this->readMarkerAdjoint != "None"){

			std::cout << "ERROR: Adjoint marker is set for the constraint function but gradient is not available!\n";
			std::cout << "Did you set GRADIENT_AVAILABLE properly?\n";
			abort();

		}

	}


	std::ifstream ifile(fileNameInputRead, std::ios::in);

	if (!ifile.is_open()) {

		std::cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}



	if(ifGradientAvailable && ifMarkerIsSet){

		if(ifAdjointMarkerIsSet== false){

			std::cout << "ERROR: Adjoint marker not is set for a constraint function!\n";
			std::cout << "Did you set CONSTRAINT_FUNCTION_GRADIENT_READ_MARKER properly?\n";
			abort();

		}



	}


	/* input from file without using markers */

	if(ifMarkerIsSet == false && ifAdjointMarkerIsSet == false ){

		double functionValue;
		ifile >> functionValue;


		d.constraintTrueValues(ID) = functionValue;


		if(ifGradientAvailable){


			for(unsigned int i=0; i<dim; i++){

				ifile >> constraintGradient(i);

			}

			if(constraintGradient.has_nan()){

				std::cout<<"ERROR: NaN in constraint gradient evaluation!\n";
				abort();

			}
		}


		d.constraintGradients.push_back(constraintGradient);



	}

	/* input from file using markers (the values are separated with ',')*/
	else{

		std::size_t found1;
		std::size_t found2;
		bool markerFound = false;
		bool markerAdjointFound = false;

		for( std::string line; getline( ifile, line ); ){ /* check each line */

			std::string bufferLine = line;

			/* first search for the constraint value marker */

			std::size_t found1 = bufferLine.find(readMarker+" ");


			if (found1!=std::string::npos){


				bufferLine = removeSpacesFromString(bufferLine);
				bufferLine.erase(0,found1+1+readMarker.size());


				d.constraintTrueValues(ID) = std::stod(bufferLine);

				markerFound = true;

			}


			if(ifGradientAvailable){

				/* check for constraint gradient marker */

				std::size_t found2 = bufferLine.find(readMarkerAdjoint+" ");
				if (found2!=std::string::npos){


					bufferLine = removeSpacesFromString(line);
					bufferLine.erase(0,found2+1+readMarkerAdjoint.size());

					vec values = getDoubleValuesFromString(bufferLine,',');
#if 0
					printVector(values,"values");
#endif
					assert(values.size() == dim);

					for(unsigned int i=0; i<dim;i++){

						constraintGradient(i) = values(i);

					}


					markerAdjointFound  = true;

				}

			}



		}
#if 0
		printVector(constraintGradient,"constraintGradient");
#endif

		d.constraintGradients.push_back(constraintGradient);

		if(ifGradientAvailable && !markerAdjointFound){

			std::cout<<"ERROR: No values can be read for the constraint gradient!\n";
			abort();


		}
		if(!markerFound){

			std::cout<<"ERROR: No values can be read for the constraint function!\n";
			abort();


		}


	}

#if 0
	d.print();
#endif

}

void ConstraintFunction::print(void) const {

	std::cout << "\n#####################################################\n";
	std::cout << "Constraint function ID = "<<ID<<"\n";
	std::cout << "Name: " << name << std::endl;
	std::cout << "Dimension: " << dim << std::endl;
	std::cout << "Type of constraint: " << inequalityType << " " <<value<< std::endl;
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


void ConstraintFunction::setParametersByDefinition(ConstraintDefinition inequalityConstraint){

	assert(!inequalityConstraint.inequalityType.empty());
	assert(!inequalityConstraint.name.empty());
	assert(inequalityConstraint.inequalityType == ">" || inequalityConstraint.inequalityType == "<");


	this->ID = inequalityConstraint.ID;

	this->executableName = inequalityConstraint.executableName;
	this->executablePath = inequalityConstraint.path;
	this->fileNameDesignVector = inequalityConstraint.designVectorFilename;
	this->fileNameInputRead = inequalityConstraint.outputFilename;
	this->inequalityType = inequalityConstraint.inequalityType;
	this->name =  inequalityConstraint.name;

	this->value = inequalityConstraint.value;


	if(!inequalityConstraint.marker.empty()){
		this->readMarker = inequalityConstraint.marker;
		this->ifMarkerIsSet = true;

	}


	if(!inequalityConstraint.markerForGradient.empty()){


		this->readMarkerAdjoint = inequalityConstraint.markerForGradient;
		this->ifAdjointMarkerIsSet = true;

	}



	ifInequalityConstraintSpecified  = true;



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
			std::cout<<"calling a system command\n";
#endif

			system(runCommand.c_str());

		}

#if 0
		std::cout<<"No need to call a system command\n";
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


