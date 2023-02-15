/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#include "bounds.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){


}


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(std::string name){

	this->name = name;

}

unsigned int ObjectiveFunctionDefinition::identifyCase(void) const{

	if(ifMultiLevel == false && ifGradient == false && ifTangent == false) return 1;
	if(ifMultiLevel == false && ifGradient == true  && ifTangent == false) return 2;
	if(ifMultiLevel == false && ifGradient == false  && ifTangent == true) return 3;

	if(ifMultiLevel == true && ifGradient == false  && ifTangent == false &&
			ifGradientLowFi == false && ifTangentLowFi == false

	) return 4;

	if(ifMultiLevel == true && ifGradient == true  && ifTangent == false &&
			ifGradientLowFi == true && ifTangentLowFi == false

	) return 5;
	if(ifMultiLevel == true && ifGradient == false  && ifTangent == true &&
			ifGradientLowFi == false && ifTangentLowFi == true

	) return 6;

	if(ifMultiLevel == true && ifGradient == false  && ifTangent == false &&
			ifGradientLowFi == true && ifTangentLowFi == false

	) return 7;


	else return 0;

}




bool ObjectiveFunctionDefinition::checkIfDefinitionIsOk(void) const{

	unsigned int idCase = identifyCase();

	if(idCase == 0){
		abortWithErrorMessage("Some problem in the objective function definition");

	}
	if(name.empty()){
		abortWithErrorMessage("Name is missing the objective function definition");
	}

	if(designVectorFilename.empty()){

		abortWithErrorMessage("Design vector filename is missing in the objective function definition");
	}

	if(executableName.empty()){
		abortWithErrorMessage("Name of the executable is missing in the objective function definition");

	}
	if(outputFilename.empty()){
		abortWithErrorMessage("Name of the output file is missing in the objective function definition");
	}

	if(nameHighFidelityTrainingData.empty()){
		abortWithErrorMessage("Name of the training data file is missing in the objective function definition");
	}

	if(idCase>3){

		if(executableNameLowFi.empty()){
			abortWithErrorMessage("Name of executable for the low fidelity model is missing in the objective function definition");

		}

		if(nameLowFidelityTrainingData.empty()){
			abortWithErrorMessage("Name of training data file for the low fidelity model is missing in the objective function definition");
		}

		if(outputFilenameLowFi.empty()){
			abortWithErrorMessage("Name of output file for the low fidelity model is missing in the objective function definition");
		}

		if(nameLowFidelityTrainingData == nameHighFidelityTrainingData){

			abortWithErrorMessage("Name of the training data is same for both low and high fidelity models");

		}


	}


	return true;

}


void ObjectiveFunctionDefinition::print(void) const{

	std::cout<<"\nObjective function definition = \n";
	std::cout<< "Name = "<<name<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";


	if(ifMultiLevel == false){


		std::cout<< "Output filename = "<<outputFilename<<"\n";
		std::cout<< "Executable name = "<<executableName<<"\n";

		if(isNotEmpty(path)){

			std::cout<< "Executable path = "<<path<<"\n";

		}

	}

	else{

		std::cout<<"Multilevel option is active...\n";


	}




}



ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, unsigned int dimension){


	dim = dimension;
	name = objectiveFunName;

}


ObjectiveFunction::ObjectiveFunction(){}

void ObjectiveFunction::setEvaluationMode(std::string mode){

	assert(isNotEmpty(mode));

	evaluationMode = mode;
}




void ObjectiveFunction::setDimension(unsigned int dimension){
	dim = dimension;
}
unsigned int ObjectiveFunction::getDimension(void) const{
	return dim;
}



void ObjectiveFunction::bindSurrogateModel(void){

	assert(ifDefinitionIsSet);

	unsigned int idCase = definition.identifyCase();

	if(idCase == 1){

		output.printMessage("Binding the surrogate model with the Kriging modeĺ...");
		surrogateModel.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModel;
	}

	if(idCase == 2){

		output.printMessage("Binding the surrogate model with the Aggregation modeĺ...");
		surrogateModelGradient.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModelGradient;
	}

	if(idCase == 3){
		output.printMessage("Binding the surrogate model with the tangent enhanced modeĺ...");
		surrogateModelWithTangents.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModelWithTangents;
	}

	if(idCase == 4){
		output.printMessage("Binding the surrogate model with the Multi-fidelity model...");

		surrogateModelML.setName(definition.name);
		surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
		surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);
		surrogate = &surrogateModelML;
	}



	if(idCase == 0){

		abortWithErrorMessage("Something is wrong with the case definition in objective function...");
	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){


	assert(def.checkIfDefinitionIsOk());
	definition = def;

	//	executableName = definition.executableName;
	//	executablePath = definition.path;
	//	fileNameDesignVector = definition.designVectorFilename;
	//	fileNameInputRead = definition.outputFilename;
	//	ifGradientAvailable = definition.ifGradient;
	//
	//
	//	if(isNotEmpty(definition.marker)){
	//
	//		readMarker = definition.marker;
	//		ifMarkerIsSet = true;
	//
	//	}
	//
	//
	//	if(isNotEmpty(definition.markerForGradient)){
	//
	//
	//		readMarkerAdjoint = definition.markerForGradient;
	//		ifAdjointMarkerIsSet = true;
	//
	//	}
	//
	//
	//	if(definition.ifMultiLevel){
	//
	//
	//		ifMultilevel = definition.ifMultiLevel;
	//		executableNameLowFi = definition.executableNameLowFi;
	//		fileNameInputReadLowFi = definition.outputFilenameLowFi;
	//		executablePathLowFi = definition.pathLowFi;
	//		readMarkerLowFi = definition.markerLowFi;
	//		readMarkerAdjointLowFi = definition.markerForGradientLowFi;
	//		fileNameTrainingDataForSurrogateHighFidelity = definition.nameHighFidelityTrainingData;
	//		fileNameTrainingDataForSurrogateLowFidelity = definition.nameLowFidelityTrainingData;
	//
	//	}


	ifDefinitionIsSet = true;



}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *)){

	objectiveFunPtr = objFun;
	ifFunctionPointerIsSet = true;

}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *, double *)){

	objectiveFunAdjPtr = objFun;
	ifFunctionPointerIsSet = true;

}




void ObjectiveFunction::setGradientOn(void){

	ifGradientAvailable = true;

}
void ObjectiveFunction::setGradientOff(void){

	ifGradientAvailable = false;

}


void ObjectiveFunction::setDisplayOn(void){

	output.ifScreenDisplay = true;
}
void ObjectiveFunction::setDisplayOff(void){

	output.ifScreenDisplay = false;

}




void ObjectiveFunction::setNumberOfTrainingIterationsForSurrogateModel(unsigned int nIter){

	numberOfIterationsForSurrogateTraining = nIter;

}

void ObjectiveFunction::setFileNameReadInput(std::string fileName){

	assert(!fileName.empty());
	definition.outputFilename = fileName;
}

void ObjectiveFunction::setFileNameReadInputLowFidelity(std::string fileName){

	assert(!fileName.empty());
	definition.outputFilenameLowFi = fileName;
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
	lowerBounds = lb;
	upperBounds = ub;

	ifParameterBoundsAreSet = true;
}


void ObjectiveFunction::setParameterBounds(Bounds bounds){

	assert(dim == bounds.getDimension());
	lowerBounds = bounds.getLowerBounds();
	upperBounds = bounds.getUpperBounds();

	ifParameterBoundsAreSet = true;


}


KrigingModel ObjectiveFunction::getSurrogateModel(void) const{

	return surrogateModel;

}

AggregationModel ObjectiveFunction::getSurrogateModelGradient(void) const{

	return surrogateModelGradient;

}

MultiLevelModel ObjectiveFunction::getSurrogateModelML(void) const{

	return surrogateModelML;

}

TGEKModel ObjectiveFunction::getSurrogateModelTangent(void) const{

	return surrogateModelWithTangents;

}




void ObjectiveFunction::setReadMarker(std::string marker){

	assert(isNotEmpty(marker));

	readMarker = marker;
	ifMarkerIsSet = true;

}
std::string ObjectiveFunction::getReadMarker(void) const{

	return readMarker;
}
void ObjectiveFunction::setReadMarkerAdjoint(std::string marker){

	assert(isNotEmpty(marker));

	readMarkerAdjoint = marker;
	ifAdjointMarkerIsSet = true;

}

std::string ObjectiveFunction::getReadMarkerAdjoint(void) const{

	return readMarkerAdjoint;
}

void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);
	assert(ifDefinitionIsSet);

	bindSurrogateModel();


	Bounds boxConstraints(lowerBounds,upperBounds);

	surrogate->setBoxConstraints(boxConstraints);


	surrogate->readData();


	surrogate->normalizeData();
	surrogate->initializeSurrogateModel();
	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

#if 0
	surrogate->printSurrogateModel();
#endif


	ifInitialized = true;

}

void ObjectiveFunction::trainSurrogate(void){

	assert(ifInitialized);
	surrogate->train();
}



void ObjectiveFunction::saveDoEData(std::vector<rowvec> data) const{

	std::string fileName = surrogate->getNameOfInputFile();

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


	surrogate->calculateExpectedImprovement(designCalculated);

}




bool ObjectiveFunction::checkIfGradientAvailable(void) const{

	return ifGradientAvailable;

}


std::string ObjectiveFunction::getExecutionCommand(void) const{

	std::string runCommand;

	if(isNotEmpty(definition.path)) {

		runCommand = definition.path +"/" + definition.executableName;
	}
	else{

		runCommand = "./" + definition.executableName;
	}

	return runCommand;


}

std::string ObjectiveFunction::getExecutionCommandLowFi(void) const{

	assert(ifMultilevel);
	assert(isNotEmpty(executableNameLowFi));

	std::string runCommand;

	if(isNotEmpty(executablePathLowFi)) {

		runCommand = executablePathLowFi +"/" + executableNameLowFi;
	}
	else{

		runCommand = "./" + executableNameLowFi;
	}

	return runCommand;

}





void ObjectiveFunction::addDesignToData(Design &d){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);

	rowvec newsample;

	if(evaluationMode.compare("primal") == 0 ){
		newsample = d.constructSampleObjectiveFunction();
	}
	if(evaluationMode.compare("tangent") == 0 ){
		newsample = d.constructSampleObjectiveFunctionWithTangent();
	}
	if(evaluationMode.compare("adjoint") == 0 ){
		newsample = d.constructSampleObjectiveFunctionWithGradient();
	}



	assert(newsample.size()>0);
	surrogate->addNewSampleToData(newsample);


}


void ObjectiveFunction::addLowFidelityDesignToData(Design &d){

	rowvec newsample;

	if(ifGradientAvailable){

		newsample = d.constructSampleObjectiveFunctionWithGradient();


	}
	else{

		newsample = d.constructSampleObjectiveFunction();

	}


	surrogate->addNewLowFidelitySampleToData(newsample);


}


rowvec ObjectiveFunction::readOutput(unsigned int howMany) const{

	assert(isNotEmpty(definition.outputFilename));

	rowvec result(howMany,fill::zeros);

	std::ifstream inputFileStream(definition.outputFilename);

	if (!inputFileStream.is_open()) {
		abortWithErrorMessage("There was a problem opening the output file!\n");
	}

	for(unsigned int i=0; i<howMany;i++){

		inputFileStream >> result(i);

	}

	inputFileStream.close();
	return result;
}


void ObjectiveFunction::readOutputDesign(Design &d) const{

	if(evaluationMode.compare("primal") == 0 ){

		rowvec functionalValue(1);

		functionalValue = readOutput(1);
		d.trueValue = functionalValue(0);
		d.objectiveFunctionValue = d.trueValue;

	}

	if(evaluationMode.compare("tangent") == 0 ){

		rowvec resultBuffer(2);

		resultBuffer = readOutput(2);
		d.trueValue = resultBuffer(0);
		d.objectiveFunctionValue = d.trueValue;

		d.tangentValue = resultBuffer(1);



	}

	if(evaluationMode.compare("adjoint") == 0 ){

		rowvec resultBuffer(1+dim);

		resultBuffer = readOutput(1+dim);
		d.trueValue = resultBuffer(0);
		d.objectiveFunctionValue = d.trueValue;

		rowvec gradient(dim,fill::zeros);

		for(unsigned int i=0; i<dim; i++){

			gradient(i) = resultBuffer(i+1);
		}

		d.gradient = gradient;

	}

}




void ObjectiveFunction::readEvaluateOutput(Design &d){



	assert(!this->fileNameInputRead.empty());
	assert(d.dimension == dim);

	if(ifMarkerIsSet && ifGradientAvailable){

		if(!ifAdjointMarkerIsSet){

			cout << "ERROR: Adjoint marker is not set for: "<< this->name<<"\n";
			abort();

		}

	}

	if(ifAdjointMarkerIsSet == true && ifGradientAvailable == false){


		cout << "ERROR: Adjoint marker is set for the objective function but gradient is not available!\n";
		cout << "Did you set GRADIENT_AVAILABLE properly?\n";
		abort();
	}


	std::ifstream ifile(fileNameInputRead, ios::in);

	if (!ifile.is_open()) {

		cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}



	for( std::string line; getline( ifile, line ); ){

		size_t found = line.find(readMarker+" ");



		if (found != std::string::npos){

			line = removeSpacesFromString(line);
			line.erase(0,found+1+this->readMarker.size());

			d.trueValue = stod(line);
			d.objectiveFunctionValue = stod(line);

		}

		if(this->ifGradientAvailable){


			size_t found2 = line.find(readMarkerAdjoint+" ");



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



	ifile.close();



}

void ObjectiveFunction::writeDesignVariablesToFile(Design &d) const{

	assert(d.designParameters.size() == dim);
	assert(isNotEmpty(definition.designVectorFilename));

	std::ofstream outputFileStream(definition.designVectorFilename);

	if (!outputFileStream.is_open()) {
		abortWithErrorMessage("There was a problem opening the output file!\n");
	}

	for(unsigned int i=0; i<dim; i++) {

		outputFileStream << d.designParameters(i) << std::endl;
	}

	if(evaluationMode.compare("tangent") == 0){

		assert(d.tangentDirection.size() == dim);
		for(unsigned int i=0; i<dim; i++) {

			outputFileStream << d.tangentDirection(i) << std::endl;
		}

	}
	outputFileStream.close();

}


void ObjectiveFunction::evaluateDesign(Design &d){

	assert(d.designParameters.size() == dim);

	writeDesignVariablesToFile(d);
	evaluateObjectiveFunction();
	readOutputDesign(d);

}


void ObjectiveFunction::evaluateObjectiveFunction(void){

	assert(isNotEmpty(definition.executableName));
	assert(isNotEmpty(definition.designVectorFilename));

	std::string runCommand = getExecutionCommand();

	output.printMessage("Calling executable for the objective function:", definition.name);
	system(runCommand.c_str());

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
		cout<<"calling a system command\n";
#endif
		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}


}



void ObjectiveFunction::evaluateLowFidelity(Design &d){

	std::string runCommand = getExecutionCommandLowFi();

#if 0
	cout<<"calling a system command\n";
#endif
	system(runCommand.c_str());


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

	return surrogate->interpolate(x);

}

void ObjectiveFunction::print(void) const{

	std::cout << "\n#####################################################\n";
	std::cout<<"Objective Function"<<endl;
	std::cout<<"Name: "<<name<<endl;
	std::cout<<"Dimension: "<<dim<<endl;
	std::cout<<"ExecutableName: "<<executableName<<"\n";
	std::cout<<"ExecutablePath: "<<executablePath<<"\n";
	std::cout<<"Output filename: "<<fileNameInputRead<<"\n";
	std::cout<<"Design vector filename: "<<fileNameDesignVector<<"\n";

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

	surrogate->printSurrogateModel();

}


