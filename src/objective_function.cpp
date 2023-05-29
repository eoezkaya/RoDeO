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
		std::cout<< "\tTraining data = " << nameLowFidelityTrainingData << "\n";;
		std::cout<< "\tExecutable = " << executableNameLowFi << "\n";
		std::cout<< "\tPath = " << path << "\n";
		std::cout<< "\tSurrogate model = " << modelLowFi << "\n";

	}


	std::cout<< "================================================================\n\n";
}


ObjectiveFunction::ObjectiveFunction(){}

void ObjectiveFunction::setEvaluationMode(std::string mode){
	assert(isNotEmpty(mode));
	evaluationMode = mode;
}


void ObjectiveFunction::setDataAddMode(std::string mode){

	assert(isNotEmpty(mode));
	addDataMode = mode;
}

void ObjectiveFunction::setDimension(unsigned int dimension){
	dim = dimension;
}
unsigned int ObjectiveFunction::getDimension(void) const{
	return dim;
}


bool ObjectiveFunction::isMultiFidelityActive(void) const{
	return definition.ifMultiLevel;
}


SURROGATE_MODEL ObjectiveFunction::getSurrogateModelType(void) const{
	return definition.modelHiFi;
}
SURROGATE_MODEL ObjectiveFunction::getSurrogateModelTypeLowFi(void) const{
	return definition.modelLowFi;
}

void ObjectiveFunction::bindWithOrdinaryKrigingModel() {
	output.printMessage(
			"Binding the surrogate model with the ORDINARY_KRIGING modeĺ...");
	surrogateModel.setNameOfInputFile(definition.nameHighFidelityTrainingData);
	surrogateModel.setName(definition.name);
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithUniversalKrigingModel() {
	output.printMessage(
			"Binding the surrogate model with the UNIVERSAL_KRIGING modeĺ...");
	surrogateModel.setNameOfInputFile(definition.nameHighFidelityTrainingData);
	surrogateModel.setLinearRegressionOn();
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithGradientEnhancedModel() {
	output.printMessage(
			"Binding the surrogate model with the GRADIENT_ENHANCED modeĺ...");
	surrogateModelGradient.setNameOfInputFile(
			definition.nameHighFidelityTrainingData);
	surrogateModelGradient.setName(definition.name);
	surrogate = &surrogateModelGradient;

}

void ObjectiveFunction::bindWithTangentEnhancedKrigingModel() {

	output.printMessage(
			"Binding the surrogate model with the TANGENT modeĺ...");
	surrogateModelWithTangents.setNameOfInputFile(
			definition.nameHighFidelityTrainingData);
	surrogate = &surrogateModelWithTangents;
}

void ObjectiveFunction::bindWithMultiFidelityModel() {

	surrogateModelML.setIDHiFiModel(definition.modelHiFi);
	surrogateModelML.setIDLowFiModel(definition.modelLowFi);

	surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
	surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);

	assert(dim>0);

	/* TODO modify this ugly code */
	surrogateModelML.setDimension(dim);
	surrogateModelML.bindModels();
	surrogateModelML.setDimension(dim);

	output.printMessage("Binding the surrogate model with the Multi-fidelity model...");

	surrogateModelML.setName(definition.name);

	surrogate = &surrogateModelML;

}

void ObjectiveFunction::bindSurrogateModelSingleFidelity() {
	if (definition.modelHiFi == ORDINARY_KRIGING) {
		bindWithOrdinaryKrigingModel();
	}
	if (definition.modelHiFi == UNIVERSAL_KRIGING) {
		bindWithUniversalKrigingModel();
	}
	if (definition.modelHiFi == GRADIENT_ENHANCED) {
		bindWithGradientEnhancedModel();
	}
	if (definition.modelHiFi == TANGENT) {
		bindWithTangentEnhancedKrigingModel();
	}
}

void ObjectiveFunction::bindSurrogateModel(void){

	assert(ifDefinitionIsSet);
	assert(definition.modelHiFi != NONE);

	if(definition.ifMultiLevel){
		bindWithMultiFidelityModel();
	}
	else{
		bindSurrogateModelSingleFidelity();
	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){


	assert(def.checkIfDefinitionIsOk());
	definition = def;
	ifDefinitionIsSet = true;

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

GGEKModel ObjectiveFunction::getSurrogateModelGradient(void) const{
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
	assert(dim>0);


	bindSurrogateModel();

	assert(boxConstraints.areBoundsSet());

	surrogate->setDimension(dim);
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



	if(ifWarmStart){
		surrogate->setReadWarmStartFileFlag(true);
	}

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

	double	sigma = sqrt(ssqr);

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

std::string ObjectiveFunction::getExecutionCommand(string path, string exename) const{

	assert(isNotEmpty(exename));

	std::string runCommand;

	if(isNotEmpty(path)) {
		runCommand = path +"/" + exename;
	}
	else{
		runCommand = "./" + exename;
	}
	return runCommand;
}




void ObjectiveFunction::addDesignToData(Design &d){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(addDataMode));


	if(definition.ifMultiLevel == false){

		rowvec newsample;

		if(addDataMode.compare("primal") == 0 ){
			newsample = d.constructSampleObjectiveFunction();
		}
		if(addDataMode.compare("tangent") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithTangent();
		}
		if(addDataMode.compare("adjoint") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithGradient();
		}

		assert(newsample.size()>0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		rowvec newsampleHiFi;
		rowvec newsampleLowFi;

		if(addDataMode.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(addDataMode.compare("primalHiFiAdjointLowFi") == 0 ){

			newsampleHiFi  = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}




	}

}

void ObjectiveFunction::addLowFidelityDesignToData(Design &d){

	assert((isNotEmpty(definition.nameLowFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(addDataMode));
	assert(definition.ifMultiLevel == true);

	rowvec newsampleLowFi;

	if(addDataMode.compare("primalLowFidelity") == 0 ){
		newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);

	}

	if(addDataMode.compare("adjointLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}

	if(addDataMode.compare("tangentLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithTangentLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}





}


rowvec ObjectiveFunction::readOutput(string filename, unsigned int howMany) const{

	assert(isNotEmpty(filename));

	rowvec result(howMany,fill::zeros);

	std::ifstream inputFileStream(filename);

	if (!inputFileStream.is_open()) {
		abortWithErrorMessage("There was a problem opening the output file!\n");
	}

	for(unsigned int i=0; i<howMany;i++){
		inputFileStream >> result(i);
	}

	inputFileStream.close();

	return result;
}

void ObjectiveFunction::readOnlyFunctionalValue(Design &d) const {
	unsigned int howManyEntriesToRead = 1;
	rowvec functionalValue(howManyEntriesToRead);


	if(isHiFiEvaluation()){
		functionalValue = readOutput(definition.outputFilename,howManyEntriesToRead);
		d.trueValue = functionalValue(0);
	}

	if(isLowFiEvaluation()){

		functionalValue = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
		d.trueValueLowFidelity = functionalValue(0);
	}


}



void ObjectiveFunction::readFunctionalValueAndTangent(Design &d) const {
	unsigned int howManyEntriesToRead = 2;
	rowvec resultBuffer(howManyEntriesToRead);


	if(isHiFiEvaluation()){
		resultBuffer = readOutput(definition.outputFilename,howManyEntriesToRead);
		d.trueValue = resultBuffer(0);
		d.tangentValue = resultBuffer(1);
	}

	if(isLowFiEvaluation()){
		resultBuffer = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
		d.trueValueLowFidelity = resultBuffer(0);
		d.tangentValueLowFidelity = resultBuffer(1);
	}

}

void ObjectiveFunction::readFunctionalValueAndAdjoint(Design &d) const {

	assert(dim >0);

	unsigned int howManyEntriesToRead = 1 + dim;
	rowvec resultBuffer(howManyEntriesToRead);

	rowvec gradient(dim, fill::zeros);
	unsigned int offset = 1;

	if(isHiFiEvaluation()){

		resultBuffer = readOutput(definition.outputFilename,howManyEntriesToRead);
		for (unsigned int i = 0; i < dim; i++) {
				gradient(i) = resultBuffer(i + offset);
		}
		d.trueValue = resultBuffer(0);
		d.gradient = gradient;

	}
	if(isLowFiEvaluation()){
		resultBuffer = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
		for (unsigned int i = 0; i < dim; i++) {
				gradient(i) = resultBuffer(i + offset);
		}
		d.trueValueLowFidelity = resultBuffer(0);
		d.gradientLowFidelity = gradient;

	}

}

void ObjectiveFunction::readOutputDesign(Design &d) const{

	if(evaluationMode.compare("primal") == 0 || evaluationMode.compare("primalLowFi") == 0){

		readOnlyFunctionalValue(d);
	}

	if(evaluationMode.compare("tangent") == 0 || evaluationMode.compare("tangentLowFi") == 0 ){

		readFunctionalValueAndTangent(d);
	}

	if(evaluationMode.compare("adjoint") == 0 || evaluationMode.compare("adjointLowFi") == 0){

		readFunctionalValueAndAdjoint(d);
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

	if(evaluationMode.compare("tangent") == 0 || evaluationMode.compare("tangentLowFi") == 0){

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


bool ObjectiveFunction::isHiFiEvaluation(void) const{

	if(evaluationMode.compare("primal") == 0 ) return true;
	if(evaluationMode.compare("adjoint") == 0 ) return true;
	if(evaluationMode.compare("tangent") == 0 ) return true;
	return false;
}

bool ObjectiveFunction::isLowFiEvaluation(void) const{

	if(evaluationMode.compare("primalLowFi") == 0 ) return true;
	if(evaluationMode.compare("adjointLowFi") == 0 ) return true;
	if(evaluationMode.compare("tangentLowFi") == 0 ) return true;
	return false;
}


void ObjectiveFunction::evaluateObjectiveFunction(void){

	assert(isNotEmpty(definition.designVectorFilename));

	std::string runCommand;
	if(isHiFiEvaluation()){

		assert(isNotEmpty(definition.executableName));

		runCommand = getExecutionCommand(definition.path, definition.executableName);

	}
	if(isLowFiEvaluation()){

		assert(isNotEmpty(definition.executableNameLowFi));
		runCommand = getExecutionCommand(definition.pathLowFi, definition.executableNameLowFi);
	}

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

void ObjectiveFunction::setSigmaFactor(double factor){

	assert(factor>0.0);
	sigmaFactor = factor;

}

