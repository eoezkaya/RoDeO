/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
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

#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"
#include "../INCLUDE/Rodeo_globals.hpp"
#include "../TestFunctions/INCLUDE/test_functions.hpp"
#include "../Optimizers/INCLUDE/optimization.hpp"
#include "../Bounds/INCLUDE/bounds.hpp"



#include "./INCLUDE/constraint_functions.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


void ConstraintDefinition::setDefinition(std::string definition){

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

	constraintName = nameBuf;
	inequalityType = typeBuf;
	value = stod(valueBuf);
}

void ConstraintDefinition::print(void) const{

	std::cout<<"Constraint ID = " << ID <<"\n";

	std::cout<<"Constraint definition = ";
	std::cout<< constraintName << " ";
	std::cout<< inequalityType << " ";
	std::cout<< value<< "\n";

}



ConstraintFunction::ConstraintFunction(){}

void ConstraintFunction::setConstraintDefinition(ConstraintDefinition definitionInput){

	definitionConstraint = definitionInput;


}




void ConstraintFunction::setID(int givenID){
	definitionConstraint.ID = givenID;
}

int ConstraintFunction::getID(void) const{
	return definitionConstraint.ID;
}

void ConstraintFunction::setInequalityType(std::string type){

	assert(type.compare(">") == 0  || type.compare("<") == 0);
	definitionConstraint.inequalityType = type;

}
std::string ConstraintFunction::getInequalityType(void) const{
	return definitionConstraint.inequalityType;
}

void ConstraintFunction::setInequalityTargetValue(double value){
	definitionConstraint.value = value;

}
double ConstraintFunction::getInequalityTargetValue(void) const{
	return definitionConstraint.value;
}


bool ConstraintFunction::checkFeasibility(double valueIn) const{

	bool result = false;
	if (definitionConstraint.inequalityType.compare("<") == 0) {
		if (valueIn < definitionConstraint.value) {
			result = true;
		}
	}
	if (definitionConstraint.inequalityType.compare(">") == 0) {
		if (valueIn > definitionConstraint.value) {
			result = true;
		}
	}
	return result;
}


void ConstraintFunction::readOutputDesign(Design &d) const{

	if(evaluationMode.compare("primal") == 0 ){

		rowvec functionalValue(1);
		functionalValue = readOutput(definition.outputFilename, 1);

		assert( int(d.constraintTrueValues.size()) > getID());


		d.constraintTrueValues(getID()) = functionalValue(0);
	}

	if(evaluationMode.compare("tangent") == 0 ){

		rowvec resultBuffer(2);

		resultBuffer = readOutput(definition.outputFilename, 2);
		d.trueValue = resultBuffer(0);
		d.tangentValue = resultBuffer(1);
	}

	if(evaluationMode.compare("adjoint") == 0 ){

		rowvec resultBuffer(1+dim);

		resultBuffer = readOutput(definition.outputFilename, 1+dim);

		int id = definitionConstraint.ID;
		assert(id>=0);

		d.constraintTrueValues(id) = resultBuffer(0);

		rowvec gradient(dim,fill::zeros);

		for(unsigned int i=0; i<dim; i++){

			gradient(i) = resultBuffer(i+1);
		}
		d.constraintGradients.push_back(gradient);
	}

}

void ConstraintFunction::evaluateDesign(Design &d){

	assert(d.designParameters.size() == dim);

	writeDesignVariablesToFile(d);
	evaluateObjectiveFunction();
	readOutputDesign(d);

}

void ConstraintFunction::addDesignToData(Design &d){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);

	assert(definitionConstraint.ID >= 0);

	rowvec newsample;


	if(evaluationMode.compare("primal") == 0 ){
		newsample = d.constructSampleConstraint(definitionConstraint.ID);
	}
	if(evaluationMode.compare("tangent") == 0 ){
		newsample = d.constructSampleConstraintWithTangent(definitionConstraint.ID);
	}
	if(evaluationMode.compare("adjoint") == 0 ){
		newsample = d.constructSampleConstraintWithGradient(definitionConstraint.ID);
	}

	assert(newsample.size()>0);
	surrogate->addNewSampleToData(newsample);


}

double ConstraintFunction::interpolate(rowvec x) const{
	if(ifFunctionExplictlyDefined){

		return functionPtr(x.memptr());
	}
	else{
		return surrogate->interpolate(x);
	}

}


pair<double, double> ConstraintFunction::interpolateWithVariance(rowvec x) const{


	pair<double, double> result;

	if(ifFunctionExplictlyDefined){

		result.first = functionPtr(x.memptr());
		result.second = 0.0;
	}
	else{
		double ftilde,sigmaSqr;
		surrogate->interpolateWithVariance(x, &ftilde, &sigmaSqr);
		result.first = ftilde;
		result.second = sqrt(sigmaSqr);
	}
	return result;
}


void ConstraintFunction::setUseExplicitFunctionOn(void){


	int ID = definitionConstraint.ID;
	functionPtr = functionVector.at(ID);
	ifFunctionExplictlyDefined = true;


}



void ConstraintFunction::print(void) const{
	definitionConstraint.print();
	definition.print();


}



