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
#include "objective_function.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, double (*objFun)(double *), unsigned int dimension)
: surrogateModel(objectiveFunName),surrogateModelGradient(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = objFun;
	objectiveFunAdjPtr = emptyAdj;
	ifGradientAvailable = false;
	ifFunctionPointerIsSet = true;

	assert(dim < 1000);


}

ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, double (*objFunAdj)(double *, double *), unsigned int dimension)
: surrogateModel(objectiveFunName),surrogateModelGradient(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = empty;
	objectiveFunAdjPtr = objFunAdj;
	ifGradientAvailable = true;
	ifFunctionPointerIsSet = true;

	assert(dim < 1000);


}



ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, unsigned int dimension)
: surrogateModel(objectiveFunName),surrogateModelGradient(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = empty;
	objectiveFunAdjPtr = emptyAdj;
	executableName = "None";
	executablePath = "None";
	fileNameObjectiveFunctionRead = "None";
	fileNameDesignVector = "None";
	ifGradientAvailable = false;
	ifFunctionPointerIsSet = false;
	assert(dim < 1000);


}


ObjectiveFunction::ObjectiveFunction(){

	dim = 0;
	name = "None";
	objectiveFunPtr = empty;
	objectiveFunAdjPtr = emptyAdj;
	executableName = "None";
	executablePath = "None";
	fileNameObjectiveFunctionRead = "None";
	fileNameDesignVector = "None";
	ifGradientAvailable = false;
	ifFunctionPointerIsSet = false;


}

void ObjectiveFunction::setGradientOn(void){

	ifGradientAvailable = true;

}
void ObjectiveFunction::setGradientOff(void){

	ifGradientAvailable = false;

}

void ObjectiveFunction::setFileNameReadObjectFunction(std::string fileName){

	fileNameObjectiveFunctionRead = fileName;

}


void ObjectiveFunction::setFileNameDesignVector(std::string fileName){

	fileNameDesignVector = fileName;

}

void ObjectiveFunction::setExecutablePath(std::string path){

	executablePath = path;

}

void ObjectiveFunction::setExecutableName(std::string exeName){

	executableName = exeName;

}

void ObjectiveFunction::trainSurrogate(void){

	if(!ifGradientAvailable){

		surrogateModel.initializeSurrogateModel();
		surrogateModel.train();

	}
	else{

		surrogateModelGradient.initializeSurrogateModel();
		surrogateModelGradient.train();

	}


}



void ObjectiveFunction::saveDoEData(std::vector<rowvec> data) const{

	std::string fileName = surrogateModel.getInputFileName();

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

double ObjectiveFunction::calculateExpectedImprovement(rowvec x) const{

	double EIValue;
	if(!ifGradientAvailable){

		EIValue = surrogateModel.calculateExpectedImprovement(x);
	}else{

		EIValue = surrogateModelGradient.calculateExpectedImprovement(x);
	}

	return EIValue;

}



bool ObjectiveFunction::checkIfGradientAvailable(void) const{

	return ifGradientAvailable;

}


std::string ObjectiveFunction::getExecutionCommand(void) const{

	std::string runCommand;
	if(executablePath!= "None") {

		runCommand = executablePath +"/" + executableName;
	}
	else{
		runCommand = "./" + executableName;
	}

	return runCommand;


}




void ObjectiveFunction::addDesignToData(Design &d){

	if(ifGradientAvailable){

		rowvec newsample = d.constructSampleObjectiveFunctionWithGradient();

		surrogateModelGradient.addNewSampleToData(newsample);

	}
	else{

		rowvec newsample = d.constructSampleObjectiveFunction();

		surrogateModel.addNewSampleToData(newsample);
	}


}


void ObjectiveFunction::readEvaluateOutput(Design &d){

	if( objectiveFunPtr == empty){
		std::ifstream ifile(fileNameObjectiveFunctionRead, std::ios::in);

		if (!ifile.is_open()) {

			std::cout << "ERROR: There was a problem opening the input file!\n";
			abort();
		}

		double functionValue;
		ifile >> functionValue;

		if(std::isnan(functionValue)){

			cout<<"ERROR: NaN as the objective function value!\n";
			abort();

		}

		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

		if(ifGradientAvailable){

			for(unsigned int i=0; i<dim;i++){
				ifile >> d.gradient(i);

			}

			if(d.gradient.has_nan()){

				cout<<"ERROR: NaN in the objective function gradients!\n";
				abort();


			}



		}


	}

}



void ObjectiveFunction::evaluate(Design &d){


	if(ifFunctionPointerIsSet){

		rowvec x= d.designParameters;
		double functionValue =  objectiveFunPtr(x.memptr());
		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

	}

	else if (fileNameObjectiveFunctionRead != "None" && executableName != "None" && fileNameDesignVector != "None"){


		std::string runCommand = getExecutionCommand();

		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}




}




void ObjectiveFunction::evaluateAdjoint(Design &d){


	assert(ifGradientAvailable);

	if( ifFunctionPointerIsSet){
		rowvec x= d.designParameters;
		rowvec xb(dim);
		xb.fill(0.0);

		double functionValue =  objectiveFunAdjPtr(x.memptr(),xb.memptr());

		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;
		d.gradient = xb;

	}

	else if (fileNameObjectiveFunctionRead != "None" && executableName != "None" && fileNameDesignVector != "None"){

		std::string runCommand = getExecutionCommand();

		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}


}


double ObjectiveFunction::ftilde(rowvec x, bool ifdebug) const{

	double result;
	if(!ifGradientAvailable){

		result = surrogateModel.interpolate(x);
	}else{

		result = surrogateModelGradient.interpolate(x, ifdebug);
	}

	return result;
}

void ObjectiveFunction::print(void) const{
	std::cout << "#####################################################\n";
	std::cout<<std::endl;
	std::cout<<"Objective Function"<<std::endl;
	std::cout<<"Name: "<<name<<std::endl;
	std::cout<<"Dimension: "<<dim<<std::endl;
	std::cout<<"ExecutableName: "<<executableName<<"\n";
	std::cout<<"ExecutablePath: "<<executablePath<<"\n";
	std::cout<<"Output filename: "<<fileNameObjectiveFunctionRead<<"\n";
	std::cout<<"Input filename: "<<fileNameDesignVector<<"\n";
	if(ifGradientAvailable){
		std::cout<<"Uses gradient vector: Yes\n";

	}
	else{
		std::cout<<"Uses gradient vector: No\n";

	}


#if 0
	surrogateModel.printSurrogateModel();
	surrogateModelGradient.printSurrogateModel();
#endif
	std::cout << "#####################################################\n";

}
