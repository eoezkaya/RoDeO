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
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"
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

	ID = 0;
	dim = 0;
	name = "None";
	pConstFun = empty;
	inequalityType = "None";
	targetValue = 0.0;
	executableName = "None";
	executablePath = "None";
	fileNameConstraintFunctionRead = "None";
	fileNameDesignVector = "None";

}

ConstraintFunction::ConstraintFunction(std::string constraintName,
		std::string constraintType, double constraintValue,
		double (*fun_ptr)(double *), unsigned int dimension, bool ifSurrogate) {

	dim = dimension;
	name = constraintName;
	pConstFun = fun_ptr;
	inequalityType = constraintType;
	executableName = "None";
	executablePath = "None";
	fileNameConstraintFunctionRead = "None";
	fileNameDesignVector = "None";

	assert(dim < 1000);
	assert(constraintType == "lt" || constraintType == "gt");

	targetValue = constraintValue;
	ifNeedsSurrogate = ifSurrogate;

	if (ifNeedsSurrogate) {

		KrigingModel temp(constraintName);
		surrogateModel = temp;

	}

}

ConstraintFunction::ConstraintFunction(std::string constraintName,
		std::string constraintType, double constraintValue,
		unsigned int dimension) :
				surrogateModel(constraintName), surrogateModelGradient(constraintName) {

	ifNeedsSurrogate = true;
	pConstFun = empty;
	targetValue = constraintValue;
	dim = dimension;
	name = constraintName;
	inequalityType = constraintType;
	executableName = "None";
	executablePath = "None";
	fileNameConstraintFunctionRead = "None";
	fileNameDesignVector = "None";

}

void ConstraintFunction::setID(int givenID) {

	ID = givenID;

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

void ConstraintFunction::saveDoEData(mat data) const {

	std::string fileName = surrogateModel.getInputFileName();
	data.save(fileName, csv_ascii);

}

void ConstraintFunction::trainSurrogate(void) {

	assert(ifNeedsSurrogate);
	surrogateModel.initializeSurrogateModel();
	surrogateModel.train();

}

double ConstraintFunction::ftilde(rowvec x) const {

	assert(ifNeedsSurrogate);

	double result = 0.0;
	if(!ifGradientAvailable){

		result = surrogateModel.interpolate(x);
	}
	else{

		result = this->surrogateModelGradient.interpolate(x);
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


void ConstraintFunction::evaluate(Design &d) {

	double functionValue = 0.0;

	rowvec x = d.designParameters;
	if (pConstFun != empty) {

		functionValue = pConstFun(x.memptr());

	}

	else if (fileNameConstraintFunctionRead != "None"
			&& executableName != "None" && fileNameDesignVector != "None") {

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

		if (ifRun) {

			bool ifSaveIsSuccessful = x.save(fileNameDesignVector, raw_ascii);

			if (!ifSaveIsSuccessful) {

				std::cout
				<< "ERROR: There was a problem while saving the design vector!\n";
				abort();

			}

			std::string runCommand;
			if (this->executablePath != "None") {

				runCommand = executablePath + "/" + executableName;
			} else {
				runCommand = "./" + executableName;
			}
#if 0
			std::cout<<runCommand<<"\n";
#endif

			system(runCommand.c_str());

		}

		int numberOfTotalItemsToRead = 1;
		int IndexOfItemToRead = 1;

		if (!IDToFunctionsShareOutputFile.empty()) {

			numberOfTotalItemsToRead = IDToFunctionsShareOutputFile.size() + 1;

			for (auto it = IDToFunctionsShareOutputFile.begin();
					it != IDToFunctionsShareOutputFile.end(); it++) {

				if (*it < int(ID)) {

					IndexOfItemToRead++;
				}

			}

		}
#if 0
		std::cout<<"numberOfTotalItemsToRead = "<<numberOfTotalItemsToRead<<"\n";
		std::cout<<"IndexOfItemToRead = "<<IndexOfItemToRead<<"\n";
#endif

		std::ifstream ifile(fileNameConstraintFunctionRead, std::ios::in);

		if (!ifile.is_open()) {

			std::cout << "ERROR: There was a problem opening the input file!\n";
			abort();
		}

		double bufferRead;
		for (int i = 0; i < numberOfTotalItemsToRead; i++) {

			ifile >> bufferRead;

			if (i == IndexOfItemToRead - 1)
				functionValue = bufferRead;

		}

	} else {

		cout
		<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
		abort();
	}

	if (std::isnan(functionValue)) {

		cout << "ERROR: NaN as the objective function value!\n";
		abort();

	}

	d.constraintTrueValues(ID-1) = functionValue;


}



//double ConstraintFunction::evaluate(rowvec x, bool ifAddToData = true) {
//
//	double functionValue = 0.0;
//
//	if (pConstFun != empty) {
//
//		functionValue = pConstFun(x.memptr());
//
//	}
//
//	else if (fileNameConstraintFunctionRead != "None"
//			&& executableName != "None" && fileNameDesignVector != "None") {
//
//		bool ifRun = true;
//		if (!IDToFunctionsShareOutputExecutable.empty()) {
//			std::vector<int>::iterator result = std::min_element(
//					IDToFunctionsShareOutputExecutable.begin(),
//					IDToFunctionsShareOutputExecutable.end());
//			unsigned int minIDshareExe = *result;
//
//			if (this->ID > minIDshareExe) {
//
//				ifRun = false;
//#if 0
//				std::cout<<"No need to call this constraint exe\n";
//#endif
//			}
//
//		}
//
//		if (ifRun) {
//
//			bool ifSaveIsSuccessful = x.save(fileNameDesignVector, raw_ascii);
//
//			if (!ifSaveIsSuccessful) {
//
//				std::cout
//				<< "ERROR: There was a problem while saving the design vector!\n";
//				abort();
//
//			}
//
//			std::string runCommand;
//			if (this->executablePath != "None") {
//
//				runCommand = executablePath + "/" + executableName;
//			} else {
//				runCommand = "./" + executableName;
//			}
//#if 0
//			std::cout<<runCommand<<"\n";
//#endif
//
//			system(runCommand.c_str());
//
//		}
//
//		int numberOfTotalItemsToRead = 1;
//		int IndexOfItemToRead = 1;
//
//		if (!IDToFunctionsShareOutputFile.empty()) {
//
//			numberOfTotalItemsToRead = IDToFunctionsShareOutputFile.size() + 1;
//
//			for (auto it = IDToFunctionsShareOutputFile.begin();
//					it != IDToFunctionsShareOutputFile.end(); it++) {
//
//				if (*it < int(ID)) {
//
//					IndexOfItemToRead++;
//				}
//
//			}
//
//		}
//#if 0
//		std::cout<<"numberOfTotalItemsToRead = "<<numberOfTotalItemsToRead<<"\n";
//		std::cout<<"IndexOfItemToRead = "<<IndexOfItemToRead<<"\n";
//#endif
//
//		std::ifstream ifile(fileNameConstraintFunctionRead, std::ios::in);
//
//		if (!ifile.is_open()) {
//
//			std::cout << "ERROR: There was a problem opening the input file!\n";
//			abort();
//		}
//
//		double bufferRead;
//		for (int i = 0; i < numberOfTotalItemsToRead; i++) {
//
//			ifile >> bufferRead;
//
//			if (i == IndexOfItemToRead - 1)
//				functionValue = bufferRead;
//
//		}
//
//	} else {
//
//		cout
//		<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
//		abort();
//	}
//
//	if (std::isnan(functionValue)) {
//
//		cout << "ERROR: NaN as the objective function value!\n";
//		abort();
//
//	}
//
//	assert(std::isnan(functionValue) == false);
//
//	if (ifAddToData) {
//
//		rowvec newsample(dim + 1);
//
//		for (unsigned int k = 0; k < dim; k++) {
//
//			newsample(k) = x(k);
//
//		}
//		newsample(dim) = functionValue;
//
//		std::cout << name << " = " << functionValue << "\n";
//
//#if 0
//		printf("new sample: \n");
//		newsample.print();
//#endif
//
//		surrogateModel.addNewSampleToData(newsample);
//
//	}
//
//	return functionValue;
//}


void ConstraintFunction::evaluateAdjoint(Design &d) const{

	assert(ifGradientAvailable);

	rowvec constraintGradient(dim);

	rowvec x = d.designParameters;
	double constraintValue = 0.0;

	if (pConstFunAdj != emptyAdj) {

		rowvec xb(dim);
		xb.fill(0.0);
		constraintValue = pConstFunAdj(x.memptr(), xb.memptr());



		for (unsigned int i = 0; i < dim; i++) {

			constraintGradient(i) = xb(i);

		}

	} else if (fileNameConstraintFunctionRead != "None"
			&& executableName != "None" && fileNameDesignVector != "None") {

		bool ifSaveIsSuccessful = x.save(fileNameDesignVector, raw_ascii);

		if (!ifSaveIsSuccessful) {

			std::cout
			<< "ERROR: There was a problem while saving the design vector!\n";
			abort();

		}

		std::string runCommand;
		if (this->executablePath != "None") {

			runCommand = executablePath + "/" + executableName;
		} else {
			runCommand = "./" + executableName;
		}
#if 0
		std::cout<<runCommand<<"\n";
#endif

		system(runCommand.c_str());

		std::ifstream ifile(fileNameConstraintFunctionRead, std::ios::in);

		if (!ifile.is_open()) {

			std::cout << "ERROR: There was a problem opening the input file!\n";
			abort();
		}

		double bufferRead;
		for (unsigned int i = 0; i < dim + 1; i++) {

			ifile >> bufferRead;

			if(i == 0){
				constraintValue = bufferRead;
			}
			else{
				constraintGradient(i-1) =  bufferRead;

			}

		}

	} else {

		cout
		<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
		abort();
	}

	if(constraintGradient.has_nan()){

			std::cout<<"ERROR: NaN in constraint gradient evaluation!\n";
			abort();


		}




	d.constraintTrueValues(ID-1) = constraintValue;
	d.constraintGradients.push_back(constraintGradient);


}



//rowvec ConstraintFunction::evaluateAdjoint(rowvec x, bool ifAddToData) {
//
//	assert(ifGradientAvailable);
//	rowvec result(dim + 1);
//	result.fill(0.0);
//	rowvec xb(dim);
//	xb.fill(0.0);
//
//	double constraintValue = 0.0;
//
//	if (pConstFunAdj != emptyAdj) {
//
//		constraintValue = pConstFunAdj(x.memptr(), xb.memptr());
//
//		result(0) = constraintValue;
//
//		for (unsigned int i = 0; i < dim; i++) {
//
//			result(i + 1) = xb(i);
//
//		}
//
//	} else if (fileNameConstraintFunctionRead != "None"
//			&& executableName != "None" && fileNameDesignVector != "None") {
//
//		bool ifSaveIsSuccessful = x.save(fileNameDesignVector, raw_ascii);
//
//		if (!ifSaveIsSuccessful) {
//
//			std::cout
//			<< "ERROR: There was a problem while saving the design vector!\n";
//			abort();
//
//		}
//
//		std::string runCommand;
//		if (this->executablePath != "None") {
//
//			runCommand = executablePath + "/" + executableName;
//		} else {
//			runCommand = "./" + executableName;
//		}
//#if 0
//		std::cout<<runCommand<<"\n";
//#endif
//
//		system(runCommand.c_str());
//
//		std::ifstream ifile(fileNameConstraintFunctionRead, std::ios::in);
//
//		if (!ifile.is_open()) {
//
//			std::cout << "ERROR: There was a problem opening the input file!\n";
//			abort();
//		}
//
//		double bufferRead;
//		for (unsigned int i = 0; i < dim + 1; i++) {
//
//			ifile >> bufferRead;
//
//			result(i) = bufferRead;
//
//		}
//
//	} else {
//
//		cout
//		<< "ERROR: Cannot evaluate the constraint function. Check settings!\n";
//		abort();
//	}
//
//	if(result.has_nan()){
//
//		std::cout<<"ERROR: NaN in constraint function evaluation!\n";
//
//	}
//
//	return result;
//
//}

void ConstraintFunction::addDesignToData(Design &d){

	rowvec newsample;

	bool ifGradientsExist;
	if(d.gradient.size() > 0){

		ifGradientsExist = true;
		assert(d.gradient.size() == dim);
	}

	if(!ifGradientsExist){

		newsample = zeros<rowvec>(dim+1);

		for(unsigned int i=0; i<dim; i++){

			newsample(i) = d.designParameters(i);


		}
		newsample(dim) = d.trueValue;
		if(surrogateModel.addNewSampleToData(newsample) !=0){

			printf("Warning: The new sample cannot be added into the training data since it is too close to a sample!\n");

		}

	}
	else{

		newsample = zeros<rowvec>(2*dim+1);


		for(unsigned int i=0; i<dim; i++){

			newsample(i) = d.designParameters(i);

		}
		newsample(dim) = d.trueValue;

		for(unsigned int i=0; i<dim; i++){


			newsample(dim+1+i) = d.gradient(i);

		}


#if 0
		printf("new sample: \n");
		newsample.print();
#endif

		if(surrogateModelGradient.addNewSampleToData(newsample) !=0){

			printf("Warning: The new sample cannot be added into the training data since it is too close to a sample!\n");

		}

	}


}



void ConstraintFunction::print(void) const {

	std::cout << "#####################################################\n";
	std::cout << std::endl;
	std::cout << "Constraint Function\n";
	std::cout << "ID: " << ID << std::endl;
	std::cout << "Name: " << name << std::endl;
	std::cout << "Dimension: " << dim << std::endl;
	std::cout << "Type of constraint: " << inequalityType << " " << targetValue
			<< std::endl;
	std::cout << "Needs surrogate:" << ifNeedsSurrogate << std::endl;
	std::cout << "Executable name: " << executableName << "\n";
	std::cout << "Executable path: " << executablePath << "\n";
	std::cout << "Input file name: " << fileNameDesignVector << "\n";
	std::cout << "Output file name: " << fileNameConstraintFunctionRead << "\n";
	std::cout << "Shares executable with:";
	for (std::vector<int>::const_iterator i =
			IDToFunctionsShareOutputExecutable.begin();
			i != IDToFunctionsShareOutputExecutable.end(); ++i)
		std::cout << " " << *i << ' ';

	std::cout << "Shares output file with:";
	for (std::vector<int>::const_iterator i =
			IDToFunctionsShareOutputFile.begin();
			i != IDToFunctionsShareOutputFile.end(); ++i)
		std::cout << " " << *i << ' ';

	std::cout << std::endl;

	surrogateModel.printSurrogateModel();
	std::cout << "#####################################################\n";
}

