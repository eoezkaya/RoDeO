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
#include "objective_function.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){


}


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(std::string name){

	this->name = name;

}


void ObjectiveFunctionDefinition::print(void) const{

	std::cout<<"\nObjective function definition = \n";
	std::cout<< "Name = "<<name<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";


	if(this->ifMultiLevel == false){


		std::cout<< "Output filename = "<<outputFilename<<"\n";
		std::cout<< "Executable name = "<<executableName<<"\n";

		if(!path.empty()){

			std::cout<< "Executable path = "<<path<<"\n";

		}

		if(!marker.empty()){

			std::cout<< "Marker = "<<marker<<"\n";


		}

	}

	else{







	}


}



ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, unsigned int dimension)
: surrogateModel(objectiveFunName),surrogateModelGradient(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = empty;
	objectiveFunAdjPtr = emptyAdj;
	assert(dim < 1000);


}


ObjectiveFunction::ObjectiveFunction(){

	objectiveFunPtr = empty;
	objectiveFunAdjPtr = emptyAdj;



}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition definition){

	this->executableName = definition.executableName;
	this->executablePath = definition.path;
	this->fileNameDesignVector = definition.designVectorFilename;
	this->fileNameInputRead = definition.outputFilename;

	if(!definition.marker.empty()){
		this->readMarker = definition.marker;
		this->ifMarkerIsSet = true;

	}


	if(!definition.markerForGradient.empty()){


		this->readMarkerAdjoint = definition.markerForGradient;
		this->ifAdjointMarkerIsSet = true;

	}






}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *)){

	this->objectiveFunPtr = objFun;
	this->ifFunctionPointerIsSet = true;

}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *, double *)){

	this->objectiveFunAdjPtr = objFun;
	this->ifFunctionPointerIsSet = true;

}




void ObjectiveFunction::setGradientOn(void){

	ifGradientAvailable = true;

}
void ObjectiveFunction::setGradientOff(void){

	ifGradientAvailable = false;

}


void ObjectiveFunction::setNumberOfTrainingIterationsForSurrogateModel(unsigned int nIter){

	numberOfIterationsForSurrogateTraining = nIter;

}

void ObjectiveFunction::setFileNameReadInput(std::string fileName){

	assert(!fileName.empty());
	fileNameInputRead = fileName;

}

void ObjectiveFunction::setFileNameDesignVector(std::string fileName){

	assert(!fileName.empty());
	fileNameDesignVector = fileName;

}

void ObjectiveFunction::setExecutablePath(std::string path){

	assert(!path.empty());
	executablePath = path;

}

void ObjectiveFunction::setExecutableName(std::string exeName){

	assert(!exeName.empty());
	executableName = exeName;

}

void ObjectiveFunction::setParameterBounds(vec lb, vec ub){

	assert(dim == lb.size());
	assert(dim == ub.size());
	this->lb = lb;
	this->ub = ub;

	ifParameterBoundsAreSet = true;
}


KrigingModel ObjectiveFunction::getSurrogateModel(void) const{

	return this->surrogateModel;

}

AggregationModel ObjectiveFunction::getSurrogateModelGradient(void) const{

	return this->surrogateModelGradient;

}


void ObjectiveFunction::setReadMarker(std::string marker){

	assert(!marker.empty());
	readMarker = marker;
	ifMarkerIsSet = true;

}
std::string ObjectiveFunction::getReadMarker(void) const{

	return readMarker;
}
void ObjectiveFunction::setReadMarkerAdjoint(std::string marker){

	assert(!marker.empty());
	readMarkerAdjoint = marker;
	ifAdjointMarkerIsSet = true;

}
std::string ObjectiveFunction::getReadMarkerAdjoint(void) const{

	return readMarkerAdjoint;
}

void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);



	if(!ifGradientAvailable){

		surrogateModel.readData();
		surrogateModel.setParameterBounds(lb,ub);
		surrogateModel.normalizeData();
		surrogateModel.initializeSurrogateModel();
		surrogateModel.setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);


	}
	else{

		surrogateModelGradient.readData();
		surrogateModelGradient.setParameterBounds(lb,ub);
		surrogateModelGradient.normalizeData();
		surrogateModelGradient.initializeSurrogateModel();
		surrogateModelGradient.setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

	}

	ifInitialized = true;

}

void ObjectiveFunction::trainSurrogate(void){

	assert(ifInitialized);


	if(!ifGradientAvailable){


		surrogateModel.train();

	}
	else{


		surrogateModelGradient.train();

	}


}



void ObjectiveFunction::saveDoEData(std::vector<rowvec> data) const{

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




void ObjectiveFunction::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{


	if(!ifGradientAvailable){

		surrogateModel.calculateExpectedImprovement(designCalculated);

	}else{

		surrogateModelGradient.calculateExpectedImprovement(designCalculated);
	}


}




bool ObjectiveFunction::checkIfGradientAvailable(void) const{

	return ifGradientAvailable;

}


std::string ObjectiveFunction::getExecutionCommand(void) const{


	assert(!executableName.empty());

	std::string runCommand;

	if(!executablePath.empty()) {

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



	assert(!this->fileNameInputRead.empty());
	assert(d.dimension == dim);
	assert(objectiveFunPtr == empty);


	if(ifMarkerIsSet && ifGradientAvailable){

		if(!ifAdjointMarkerIsSet){

			std::cout << "ERROR: Adjoint marker is not set for: "<< this->name<<"\n";
			abort();

		}

	}

	if(ifAdjointMarkerIsSet == true && ifGradientAvailable == false){


		std::cout << "ERROR: Adjoint marker is set for the objective function but gradient is not available!\n";
		std::cout << "Did you set GRADIENT_AVAILABLE properly?\n";
		abort();
	}


	std::ifstream ifile(fileNameInputRead, std::ios::in);

	if (!ifile.is_open()) {

		std::cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}

	if(ifMarkerIsSet == false){

		/* If there is not any marker, just reads the functional value (and gradient) from the input file */

		double functionValue;
		ifile >> functionValue;


		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

		if(ifGradientAvailable){

			for(unsigned int i=0; i<dim;i++){

				ifile >> d.gradient(i);

			}

		}

	}


	else{

		for( std::string line; getline( ifile, line ); ){

			std::size_t found = line.find(readMarker+" ");



			if (found!=std::string::npos){

				line = removeSpacesFromString(line);
				line.erase(0,found+1+this->readMarker.size());

				d.trueValue = std::stod(line);
				d.objectiveFunctionValue = std::stod(line);

			}

			if(this->ifGradientAvailable){


				std::size_t found2 = line.find(readMarkerAdjoint+" ");



				if (found2!=std::string::npos){
					line = removeSpacesFromString(line);
					line.erase(0,found2+1+readMarkerAdjoint.size());
					vec values = getDoubleValuesFromString(line,',');
					assert(values.size() == dim);


					for(unsigned int i=0; i<dim; i++){

						d.gradient(i) = values(i);
					}


				}




			}
		}

	}

	ifile.close();



}



void ObjectiveFunction::evaluate(Design &d){

	if(ifFunctionPointerIsSet){

		rowvec x= d.designParameters;
		double functionValue =  objectiveFunPtr(x.memptr());
		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

	}

	else if (executableName != "None" && fileNameDesignVector != "None"){


		std::string runCommand = getExecutionCommand();

#if 0
		std::cout<<"calling a system command\n";
#endif
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

	else if (executableName != "None" && fileNameDesignVector != "None"){

		std::string runCommand = getExecutionCommand();

		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}


}


double ObjectiveFunction::interpolate(rowvec x) const{

	double result;
	if(!ifGradientAvailable){

		result = surrogateModel.interpolate(x);
	}else{

		result = surrogateModelGradient.interpolate(x);
	}


	return result;
}

void ObjectiveFunction::print(void) const{
	std::cout << "\n#####################################################\n";
	std::cout<<"Objective Function"<<std::endl;
	std::cout<<"Name: "<<name<<std::endl;
	std::cout<<"Dimension: "<<dim<<std::endl;
	std::cout<<"ExecutableName: "<<executableName<<"\n";
	std::cout<<"ExecutablePath: "<<executablePath<<"\n";
	std::cout<<"Output filename: "<<fileNameInputRead<<"\n";
	std::cout<<"Input filename: "<<fileNameDesignVector<<"\n";

	if(this->ifMarkerIsSet){

		std::cout<<"Read marker: "<<readMarker<<"\n";

	}



	if(ifGradientAvailable){
		std::cout<<"Uses gradient vector: Yes\n";


		if(this->ifAdjointMarkerIsSet){

			std::cout<<"Read marker for gradient: "<<readMarkerAdjoint<<"\n";


		}


	}
	else{
		std::cout<<"Uses gradient vector: No\n";

	}


	std::cout << "#####################################################\n\n";

}


void ObjectiveFunction::printSurrogate(void) const{

	if(ifGradientAvailable){


		surrogateModelGradient.printSurrogateModel();

	}
	else{

		surrogateModel.printSurrogateModel();

	}



}

