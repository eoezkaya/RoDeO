/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
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

ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){}

bool ObjectiveFunctionDefinition::checkIfDefinitionIsOk(void) const{


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

	if(ifMultiLevel){

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

	std::cout<<"\n================ Objective function definition ================\n";
	std::cout<< "Name = "<<name<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";
	std::cout<< "Training data = " << nameHighFidelityTrainingData << "\n";;
	std::cout<< "Executable = " << executableName << "\n";
	std::cout<< "Path = " << path << "\n";
	std::cout<< "Surrogate model = " << modelHiFi << "\n";
	std::cout<< "Multilevel = "<<ifMultiLevel<<"\n";

	if(ifMultiLevel){
		std::cout<< "Low fidelity model = " << "\n";
		std::cout<< "\tTraining data = " << nameHighFidelityTrainingData << "\n";;
		std::cout<< "\tExecutable = " << executableName << "\n";
		std::cout<< "\tPath = " << path << "\n";
		std::cout<< "\tSurrogate model = " << modelHiFi << "\n";

	}


	std::cout<< "================================================================\n\n";
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
	assert(definition.modelHiFi != NONE);


	if(definition.modelHiFi == ORDINARY_KRIGING){

		output.printMessage("Binding the surrogate model with the ORDINARY_KRIGING modeĺ...");
		surrogateModel.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModel;
	}

	if(definition.modelHiFi == UNIVERSAL_KRIGING){

		output.printMessage("Binding the surrogate model with the UNIVERSAL_KRIGING modeĺ...");
		surrogateModel.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogateModel.setLinearRegressionOn();
		surrogate = &surrogateModel;
	}

	if(definition.modelHiFi == AGGREGATION){

		output.printMessage("Binding the surrogate model with the Aggregation modeĺ...");
		surrogateModelGradient.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModelGradient;


	}

	if(definition.modelHiFi == TANGENT){
		output.printMessage("Binding the surrogate model with the tangent enhanced modeĺ...");
		surrogateModelWithTangents.setNameOfInputFile(definition.nameHighFidelityTrainingData);
		surrogate = &surrogateModelWithTangents;
	}

	if(definition.ifMultiLevel){

		if(definition.modelHiFi == ORDINARY_KRIGING && definition.modelLowFi == ORDINARY_KRIGING){

			output.printMessage("Binding the surrogate model with the Multi-fidelity model...");

			surrogateModelML.setName(definition.name);
			surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
			surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);
			surrogate = &surrogateModelML;

		}

	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){


	assert(def.checkIfDefinitionIsOk());
	definition = def;
	ifDefinitionIsSet = true;

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
	definition.designVectorFilename = fileName;
}

std::string ObjectiveFunction::getFileNameDesignVector(void) const{
	return definition.designVectorFilename;
}

std::string ObjectiveFunction::getFileNameTrainingData(void) const{
	return definition.nameHighFidelityTrainingData;
}



void ObjectiveFunction::setExecutablePath(std::string path){

	assert(!path.empty());
	definition.path = path;
}
void ObjectiveFunction::setExecutableName(std::string exeName){

	assert(!exeName.empty());
	definition.executableName = exeName;

}


void ObjectiveFunction::setParameterBounds(Bounds bounds){

	assert(dim == bounds.getDimension());
	assert(bounds.areBoundsSet());

	boxConstraints = bounds;

	if(ifSurrogateModelIsDefined){

		surrogate->setBoxConstraints(boxConstraints);
	}


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

void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);
	assert(ifDefinitionIsSet);

	bindSurrogateModel();

	assert(boxConstraints.areBoundsSet());

	surrogate->setBoxConstraints(boxConstraints);
	surrogate->readData();
	surrogate->normalizeData();
	surrogate->initializeSurrogateModel();
	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

#if 0
	surrogate->printSurrogateModel();
#endif

	yMin = min(surrogate->gety());

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

void ObjectiveFunction::calculateExpectedImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double	sigma = sqrt(ssqr)	;

#if 0
	printf("standart_ERROR = %15.10f\n",sigma);
#endif

	double expectedImprovementValue = 0.0;

	if(fabs(sigma) > EPSILON){

		double improvement = 0.0;
		improvement = yMin   - ftilde;

		double	Z = (improvement)/sigma;
#if 0
		printf("Z = %15.10f\n",Z);
		printf("ymin = %15.10f\n",yMin);
#endif

		expectedImprovementValue = improvement*cdf(Z,0.0,1.0)+  sigma * pdf(Z,0.0,1.0);


	}
	else{

		expectedImprovementValue = 0.0;

	}


	designCalculated.valueAcqusitionFunction = expectedImprovementValue;
	designCalculated.objectiveFunctionValue = ftilde;
	designCalculated.sigma = sigma;
}


void ObjectiveFunction::calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double	sigma = sqrt(ssqr)	;
	designCalculated.sigma = sigma;

	double PI = designCalculated.calculateProbalityThatTheEstimateIsLessThanAValue(yMin);

	designCalculated.valueAcqusitionFunction = PI;

	designCalculated.objectiveFunctionValue = ftilde;

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

	assert(definition.ifMultiLevel);
	assert(isNotEmpty(definition.executableNameLowFi));

	std::string runCommand;
	if(isNotEmpty(definition.pathLowFi)) {
		runCommand = definition.pathLowFi +"/" + definition.executableNameLowFi;
	}
	else{
		runCommand = "./" + definition.executableNameLowFi;
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
	}

	if(evaluationMode.compare("tangent") == 0 ){

		rowvec resultBuffer(2);

		resultBuffer = readOutput(2);
		d.trueValue = resultBuffer(0);
		d.tangentValue = resultBuffer(1);

	}

	if(evaluationMode.compare("adjoint") == 0 ){

		rowvec resultBuffer(1+dim);

		resultBuffer = readOutput(1+dim);
		d.trueValue = resultBuffer(0);
		rowvec gradient(dim,fill::zeros);

		for(unsigned int i=0; i<dim; i++){

			gradient(i) = resultBuffer(i+1);
		}
		d.gradient = gradient;

	}

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
double ObjectiveFunction::interpolate(rowvec x) const{
	return surrogate->interpolate(x);
}

pair<double, double> ObjectiveFunction::interpolateWithVariance(rowvec x) const{

	double ftilde,sigmaSqr;
	surrogate->interpolateWithVariance(x, &ftilde, &sigmaSqr);

	pair<double, double> result;
	result.first = ftilde;
	result.second = sqrt(sigmaSqr);

	return result;
}


void ObjectiveFunction::print(void) const{
	definition.print();
}

void ObjectiveFunction::printSurrogate(void) const{
	surrogate->printSurrogateModel();
}

void ObjectiveFunction::reduceTrainingDataFiles(unsigned howManySamples, double targetValue) const{

	surrogate->reduceTrainingData(howManySamples,targetValue);
}



