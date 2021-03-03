/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

ConstraintFunction::ConstraintFunction() {

	pConstFun = empty;
	pConstFunAdj = emptyAdj;


}

ConstraintFunction::ConstraintFunction(std::string constraintName,
		std::string constraintType, double constraintValue,
		double (*fun_ptr)(double *), unsigned int dimension) {

	dim = dimension;
	name = constraintName;
	pConstFun = fun_ptr;
	pConstFunAdj = emptyAdj;
	ifFunctionPointerIsSet = true;
	inequalityType = constraintType;

	assert(dim < 1000);
	assert(constraintType == "lt" || constraintType == "gt");

	targetValue = constraintValue;



}

ConstraintFunction::ConstraintFunction(std::string constraintName,
		std::string constraintType, double constraintValue,
		double (*fun_ptr)(double *, double *), unsigned int dimension) {

	dim = dimension;
	name = constraintName;
	pConstFun = empty;
	pConstFunAdj = fun_ptr;
	ifGradientAvailable = true;
	ifFunctionPointerIsSet = true;
	inequalityType = constraintType;


	assert(dim < 1000);
	assert(constraintType == "lt" || constraintType == "gt");

	targetValue = constraintValue;



}



ConstraintFunction::ConstraintFunction(std::string constraintName,
		std::string constraintType, double constraintValue,
		unsigned int dimension) :
								surrogateModel(constraintName), surrogateModelGradient(constraintName) {


	pConstFun = empty;
	pConstFunAdj = emptyAdj;
	targetValue = constraintValue;
	dim = dimension;
	name = constraintName;
	inequalityType = constraintType;



}

void ConstraintFunction::setID(unsigned int givenID) {

	ID = givenID;

}

unsigned int ConstraintFunction::getID(void) const {

	return ID;

}

void ConstraintFunction::setGradientOn(void){

	ifGradientAvailable = true;

}
void ConstraintFunction::setGradientOff(void){

	ifGradientAvailable = false;

}

void ConstraintFunction::setFileNameReadConstraintFunction(
		std::string fileName) {

	fileNameConstraintFunctionRead = fileName;

}

void ConstraintFunction::setFileNameDesignVector(std::string fileName) {

	fileNameDesignVector = fileName;

}

void ConstraintFunction::setExecutablePath(std::string path) {

	executablePath = path;

}

void ConstraintFunction::setExecutableName(std::string exeName) {

	executableName = exeName;

}

void ConstraintFunction::saveDoEData(std::vector<rowvec> data) const{

	std::string fileName = surrogateModel.getNameOfInputFile();

	std::ofstream myFile(fileName);

	myFile.precision(10);

	for(unsigned int i = 0; i < data.size(); ++i)
	{
		rowvec v=  data.at(i);


		for(unsigned int j= 0; j<v.size(); j++){

			myFile <<v(j)<<",";
		}

		myFile << "\n";
	}


	myFile.close();


}

void ConstraintFunction::setParameterBounds(vec lb, vec ub){

	this->lb = lb;
	this->ub = ub;


}

void ConstraintFunction::initializeSurrogate(void){

	if(!ifGradientAvailable){

		this->surrogateModel.readData();
		this->surrogateModel.setParameterBounds(lb,ub);
		this->surrogateModel.normalizeData();

	}
	else{


		this->surrogateModelGradient.readData();
		this->surrogateModelGradient.setParameterBounds(lb,ub);
		this->surrogateModelGradient.normalizeData();

	}


	ifInitialized = true;
}

void ConstraintFunction::trainSurrogate(void) {

	assert(ifInitialized);


	if(!ifGradientAvailable){

		surrogateModel.initializeSurrogateModel();
		surrogateModel.train();

	}
	else{

		surrogateModelGradient.initializeSurrogateModel();
		surrogateModelGradient.train();

	}




}

double ConstraintFunction::ftilde(rowvec x) const {



	double result = 0.0;
	if(!ifGradientAvailable){

		result = surrogateModel.interpolate(x);
	}
	else{

		result = surrogateModelGradient.interpolate(x);
	}
	return result;

}

bool ConstraintFunction::checkFeasibility(double value) const{

	bool result = true;
	if (inequalityType == "lt" || inequalityType == "<") {

		if (value >= targetValue) {

			result = false;

		}

	}

	if (inequalityType == "gt" || inequalityType == ">") {

		if (value <= targetValue) {

			result = false;

		}

	}

	return result;
}


bool ConstraintFunction::checkIfGradientAvailable(void) const{

	return ifGradientAvailable;

}




void ConstraintFunction::readEvaluateOutput(Design &d) {


	unsigned int totalNumberOfEntriesToRead;
	if(ifGradientAvailable){

		totalNumberOfEntriesToRead = readOutputStartIndex + dim+1;
	}
	else{

		totalNumberOfEntriesToRead = readOutputStartIndex+1;

	}

	std::ifstream ifile(fileNameConstraintFunctionRead, std::ios::in);

	if (!ifile.is_open()) {

		std::cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}

	vec bufferRead(totalNumberOfEntriesToRead);

	for (unsigned int i = 0; i < totalNumberOfEntriesToRead ; i++) {

		ifile >> bufferRead(i);
	}

	printVector(bufferRead);

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


bool ConstraintFunction::checkIfRunExecutableNecessary(void) {

	bool ifRun = true;

	if (!IDToFunctionsShareOutputExecutable.empty()) {


		std::vector<int>::iterator result = std::min_element(
				IDToFunctionsShareOutputExecutable.begin(),
				IDToFunctionsShareOutputExecutable.end());
		unsigned int minIDshareExe = *result;

		if (this->ID > minIDshareExe) {

			ifRun = false;
#if 0
			std::cout<<"No need to call this constraint exe\n";
#endif
		}

	}


	return ifRun;

}


void ConstraintFunction::evaluate(Design &d) {

	double functionValue = 0.0;

	rowvec x = d.designParameters;
	if (pConstFun != empty) {

		functionValue = pConstFun(x.memptr());




		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

	}

	else if (fileNameConstraintFunctionRead != "None"
			&& executableName != "None" && fileNameDesignVector != "None") {

		bool ifRun = checkIfRunExecutableNecessary();

		if (ifRun) {

			std::string runCommand = getExecutionCommand();


			system(runCommand.c_str());

		}





	} else {

		cout<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
		abort();
	}


}

void ConstraintFunction::evaluateAdjoint(Design &d) {

	assert(ifGradientAvailable);


	if (pConstFunAdj != emptyAdj) {
		double constraintValue = 0.0;
		rowvec xb(dim);
		xb.fill(0.0);
		rowvec x = d.designParameters;
		constraintValue = pConstFunAdj(x.memptr(), xb.memptr());

		rowvec constraintGradient = xb;

		if(constraintGradient.has_nan()){

			std::cout<<"ERROR: NaN in constraint gradient evaluation!\n";
			abort();


		}


		d.constraintTrueValues(ID-1) = constraintValue ;
		d.constraintGradients.push_back(constraintGradient);




	} else if (fileNameConstraintFunctionRead != "None"
			&& executableName != "None" && fileNameDesignVector != "None") {

		bool ifRun = checkIfRunExecutableNecessary();


		if (ifRun) {

			std::string runCommand = getExecutionCommand();

			system(runCommand.c_str());

			system(runCommand.c_str());


		}



	} else {

		cout<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
		abort();
	}



	d.print();

}


void ConstraintFunction::addDesignToData(Design &d){

	if(ifGradientAvailable){

		rowvec newsample = d.constructSampleConstraintWithGradient(ID);

#if 0
		printf("new sample: \n");
		newsample.print();
#endif

		surrogateModelGradient.addNewSampleToData(newsample);

	}
	else{

		rowvec newsample = d.constructSampleConstraint(ID);

		surrogateModel.addNewSampleToData(newsample);


	}


}

std::string ConstraintFunction::getExecutionCommand(void) const{

	std::string runCommand;
	if(!executablePath.empty()) {

		runCommand = executablePath +"/" + executableName;
	}
	else{
		runCommand = "./" + executableName;
	}

	return runCommand;


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
	std::cout << "Output file name: " << fileNameConstraintFunctionRead << "\n";
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


ConstraintFunctionv2::ConstraintFunctionv2(std::string name, unsigned int dimension): ObjectiveFunction(name, dimension){



}
ConstraintFunctionv2::ConstraintFunctionv2(std::string name, double (*objFun)(double *), unsigned int dimension): ObjectiveFunction(name, objFun,dimension){



}

void ConstraintFunctionv2::readEvaluateOutput(Design &d) {


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

void ConstraintFunctionv2::print(void) const {

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


void ConstraintFunctionv2::setInequalityConstraint(std::string inequalityStatement){

		inequalityStatement = removeSpacesFromString(inequalityStatement);

		char type = inequalityStatement.front();
		assert(type == '<' || type == '>');
		inequalityStatement.erase(0,1);
		inequalityType = type;
		double value = std::stod(inequalityStatement);
		this->targetValue = value;

		ifInequalityConstraintSpecified  = true;

	}


bool ConstraintFunctionv2::checkFeasibility(double value) const{

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



