#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>

#include "auxiliary_functions.hpp"
#include "print.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "bounds.hpp"



#include "constraint_functions.hpp"
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


void ConstraintFunction::trainSurrogate(void){

	if(!ifFunctionExplictlyDefined){
		ObjectiveFunction::trainSurrogate();
	}
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

	rowvec functionalValue(1);
	functionalValue = readOutput(definition.outputFilename, 1);

	assert( int(d.constraintTrueValues.size()) > getID());

	d.constraintTrueValues(getID()) = functionalValue(0);

}

void ConstraintFunction::evaluateDesign(Design &d){


	if(ifFunctionExplictlyDefined){

		rowvec x = d.designParameters;
		double functionValue = functionPtr(x.memptr());
		printScalar(functionValue);
		d.constraintTrueValues(getID()) = functionValue;

	}
	else{


		assert(d.designParameters.size() == dim);
		setEvaluationMode("primal");

		if(!doesObjectiveFunctionPtrExist){

			writeDesignVariablesToFile(d);
			evaluateObjectiveFunction();
			rowvec functionalValue(1);
			functionalValue = readOutput(definition.outputFilename, 1);

			assert( int(d.constraintTrueValues.size()) > getID());
			d.constraintTrueValues(getID()) = functionalValue(0);


		}
		else{

			double constraintValue = evaluateObjectiveFunctionDirectly(d.designParameters);
			assert( int(d.constraintTrueValues.size()) > getID());
			d.constraintTrueValues(getID()) = constraintValue;


		}




	}



}

void ConstraintFunction::addDesignToData(Design &d){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);

	assert(definitionConstraint.ID >= 0);

	rowvec newsample;
	newsample = d.constructSampleConstraint(definitionConstraint.ID);

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
		result.second = EPSILON;
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
	assert(ID >=0 && ID<20);

	functionPtr = functionVector.at(ID);
	ifFunctionExplictlyDefined = true;


}



void ConstraintFunction::print(void) const{

	std::cout<<"\n================= Constraint function definition =========================\n";

	definitionConstraint.print();

	if(!ifFunctionExplictlyDefined){
		definition.print();
	}
	std::cout<< "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
	std::cout<<"\n==========================================================================\n";

}

string ConstraintFunction::generateOutputString(void) const{


	std::string outputMsg;
	string tag = "Constraint function definition";
	outputMsg = generateFormattedString(tag,'=', 100) + "\n";
	outputMsg+= "Name : " + definition.name + "\n";

	if(!definition.executableName.empty()){
		outputMsg+= "Executable : " + definition.executableName + "\n";
	}

	if(doesObjectiveFunctionPtrExist){

		outputMsg+= "API Call : YES\n";
	}


	outputMsg+= "Training data : " + definition.nameHighFidelityTrainingData + "\n";

	if(!definition.outputFilename.empty()){

		outputMsg+= "Output file : " + definition.outputFilename + "\n";
	}
	if(!definition.designVectorFilename.empty()){

		outputMsg+= "Design parameters file : " + definition.designVectorFilename + "\n";
	}

	string modelName = definition.getNameOfSurrogateModel(definition.modelHiFi);
	outputMsg+= "Surrogate model : " + modelName + "\n";

	outputMsg+= "Number of iterations for model training : " +std::to_string(numberOfIterationsForSurrogateTraining) + "\n";


	std::string border(100, '=');
	outputMsg += border + "\n";


	return outputMsg;

}



bool ConstraintFunction::isUserDefinedFunction(void) const{

	return ifFunctionExplictlyDefined;
}


double ConstraintFunction::callUserDefinedFunction(rowvec &x) const{

	return functionPtr(x.memptr());

}

