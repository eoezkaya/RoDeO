/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"
#include "../INCLUDE/Rodeo_globals.hpp"
#include "../TestFunctions/INCLUDE/test_functions.hpp"
#include "../Optimizers/INCLUDE/optimization.hpp"
#include "./INCLUDE/objective_function.hpp"


#include "../Bounds/INCLUDE/bounds.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){}

bool ObjectiveFunctionDefinition::checkIfDefinitionIsOk(void) const{

	if(ifDefined == false) {

		//		std::cout<<"ifDefined is false!";
		return false;
	}

	if(name.empty()){

		std::cout<<"name is empty!";
		return false;
	}


	if(designVectorFilename.empty()){
		std::cout<<"designVectorFilename is empty!";
		return false;

	}

	if(outputFilename.empty()){

		std::cout<<"outputFilename is empty!";
		return false;
	}
	if(nameHighFidelityTrainingData.empty()){
		std::cout<<"nameHighFidelityTrainingData is empty!";
		return false;
	}

	if(ifMultiLevel){

		if(nameLowFidelityTrainingData.empty() ||
				outputFilenameLowFi.empty()){
			return false;
		}

		if(nameLowFidelityTrainingData == nameHighFidelityTrainingData){

			return false;
		}

	}
	return true;
}

string ObjectiveFunctionDefinition::getNameOfSurrogateModel(SURROGATE_MODEL modelType) const {

	string modelName;
	if (modelType == ORDINARY_KRIGING)
		modelName = "ORDINARY_KRIGING";

	if (modelType == GRADIENT_ENHANCED)
		modelName = "GRADIENT_ENHANCED";

	if (modelType == TANGENT_ENHANCED)
		modelName = "TANGENT_ENHANCED";

	return modelName;
}

void ObjectiveFunctionDefinition::printHighFidelityModel() const {

	string modelName = getNameOfSurrogateModel(modelHiFi);
	std::cout << "Surrogate model = " << modelName << "\n";

}

void ObjectiveFunctionDefinition::printLowFidelityModel() const {
	string modelNameLowFi = getNameOfSurrogateModel(modelLowFi);
	std::cout << "\tSurrogate model = " << modelNameLowFi << "\n";
}

void ObjectiveFunctionDefinition::print(void) const{

	std::cout<<"\n================ Objective/Constraint function definition ================\n";
	std::cout<< "Name = "<<name<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";
	std::cout<< "Training data = " << nameHighFidelityTrainingData << "\n";
	std::cout<< "Output data = " << outputFilename << "\n";
	std::cout<< "Executable = " << executableName << "\n";

	if(isNotEmpty(executableNameGradient)){

		std::cout<< "Executable for gradient = " << executableNameGradient << "\n";
		std::cout<< "Output file name for gradient = " << outputGradientFilename << "\n";
	}

	printHighFidelityModel();

	string ifMultiFidelity;
	if(ifMultiLevel){

		ifMultiFidelity = "YES";
	}
	else{

		ifMultiFidelity = "NO";
	}


	std::cout<< "Multilevel = "<< ifMultiFidelity <<"\n";

	if(ifMultiLevel){

		std::cout<< "Low fidelity model = " << "\n";
		std::cout<< "\tTraining data = " << nameLowFidelityTrainingData << "\n";
		std::cout<< "Output data = " << outputFilenameLowFi << "\n";
		std::cout<< "\tExecutable = " << executableNameLowFi << "\n";
		std::cout<< "\tPath = " << path << "\n";

		printLowFidelityModel();
	}


	std::cout<< "================================================================\n\n";
}



/**********************************************************************************************/


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
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithUniversalKrigingModel() {
	output.printMessage(
			"Binding the surrogate model with the UNIVERSAL_KRIGING modeĺ...");
	surrogateModel.setLinearRegressionOn();
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithGradientEnhancedModel() {
	output.printMessage(
			"Binding the surrogate model with the GRADIENT_ENHANCED modeĺ...");
	surrogateModelGradient.setAdjustThetaFactorOff();
	surrogate = &surrogateModelGradient;

}

void ObjectiveFunction::bindWithTangentEnhancedModel() {

	output.printMessage(
			"Binding the surrogate model with the TANGENT_ENHANCED modeĺ...");
	surrogateModelGradient.setDirectionalDerivativesOn();
	surrogate = &surrogateModelGradient;
}

void ObjectiveFunction::bindWithMultiFidelityModel() {

	surrogateModelML.setIDHiFiModel(definition.modelHiFi);
	surrogateModelML.setIDLowFiModel(definition.modelLowFi);

	surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
	surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);


	/* TODO modify this ugly code */
	surrogateModelML.bindModels();

	output.printMessage("Binding the surrogate model with the Multi-fidelity model...");

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
	if (definition.modelHiFi == TANGENT_ENHANCED) {
		bindWithTangentEnhancedModel();
	}
}

void ObjectiveFunction::bindSurrogateModel(void){

	assert(ifDefinitionIsSet);


	if(definition.ifMultiLevel){
		bindWithMultiFidelityModel();
	}
	else{
		bindSurrogateModelSingleFidelity();
	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){


	bool ifDefinitionIsOk = def.checkIfDefinitionIsOk();

	if(!ifDefinitionIsOk){
		def.print();
		abortWithErrorMessage("Something wrong with the objective function definition");
	}

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

mat ObjectiveFunction::getTrainingData(void) const{

	assert(ifSurrogateModelIsDefined);
	return surrogate->getRawData();

}

std::string ObjectiveFunction::getName(void) const{
	assert(ifDefinitionIsSet);
	return definition.name;
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

	if(dim > 0 ){
		assert(dim == bounds.getDimension());
	}


	assert(bounds.areBoundsSet());
	boxConstraints = bounds;

	ifParameterBoundsAreSet = true;
}

MultiLevelModel ObjectiveFunction::getSurrogateModelML(void) const{
	return surrogateModelML;
}



void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);
	assert(ifDefinitionIsSet);
	assert(dim>0);

	bindSurrogateModel();


	surrogate->setName(definition.name);
	surrogate->setDimension(dim);
	surrogate->setBoxConstraints(boxConstraints);
	surrogate->setNameOfInputFile(definition.nameHighFidelityTrainingData);


	surrogate->readData();
	surrogate->normalizeData();

	surrogate->initializeSurrogateModel();
	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

#if 0
	surrogate->printSurrogateModel();
#endif



	ifInitialized = true;
}

void ObjectiveFunction::setFeasibleMinimum(double value){

	sampleMinimum = value;

}


void ObjectiveFunction::trainSurrogate(void){


	assert(ifInitialized);


	if(ifWarmStart){
		surrogate->setReadWarmStartFileFlag(true);
	}

	surrogate->train();
}





void ObjectiveFunction::calculateExpectedImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double sigma = sqrt(ssqr);

	designCalculated.sigma = sigma;

	sigma = sigmaFactor*sigma;

	/* larger sigma means model uncertainty is higher. In this case, more exploration will take place */

#if 0
	printf("standart_ERROR = %15.10f\n",sigma);
#endif

	double expectedImprovementValue = 0.0;

	if(fabs(sigma) > EPSILON){

		double improvement = 0.0;
		improvement = sampleMinimum   - ftilde;

		double	Z = (improvement)/sigma;
#if 0
		printf("Z = %15.10f\n",Z);
		printf("ymin = %15.10f\n",yMin);
#endif

		expectedImprovementValue = improvement*cdf(Z,0.0,1.0)+   sigma * pdf(Z,0.0,1.0);


	}
	else{

		expectedImprovementValue = 0.0;

	}


	designCalculated.valueAcqusitionFunction = expectedImprovementValue;
	designCalculated.objectiveFunctionValue = ftilde;

}


void ObjectiveFunction::calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double	sigma = sqrt(ssqr)	;
	designCalculated.sigma = sigma;

	double PI = designCalculated.calculateProbalityThatTheEstimateIsLessThanAValue(sampleMinimum);

	designCalculated.valueAcqusitionFunction = PI;

	designCalculated.objectiveFunctionValue = ftilde;

}

void ObjectiveFunction::calculateSurrogateEstimate(DesignForBayesianOptimization &designCalculated) const{

	double ftilde;
	ftilde = surrogate->interpolate(designCalculated.dv);
	designCalculated.objectiveFunctionValue = ftilde;
}

void ObjectiveFunction::calculateSurrogateEstimateUsingDerivatives(DesignForBayesianOptimization &designCalculated) const{

	double ftilde;
	ftilde = surrogate->interpolateUsingDerivatives(designCalculated.dv);
	designCalculated.objectiveFunctionValue = ftilde;
}

std::string ObjectiveFunction::getExecutionCommand(string exename) const{

	assert(isNotEmpty(exename));
	std::string runCommand;

	runCommand = "./" + exename;
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
		if(addDataMode.compare("adjointWithZeroGradient") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithZeroGradient();
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


void ObjectiveFunction::addDesignToData(Design &d, string how){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(how));


	if(definition.ifMultiLevel == false){

		rowvec newsample;

		if(how.compare("primal") == 0 ){
			newsample = d.constructSampleObjectiveFunction();
		}
		if(how.compare("tangent") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithTangent();
		}
		if(how.compare("adjoint") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithGradient();
		}
		if(how.compare("adjointWithZeroGradient") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithZeroGradient();
		}


		assert(newsample.size()>0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		rowvec newsampleHiFi;
		rowvec newsampleLowFi;

		if(how.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(how.compare("primalHiFiAdjointLowFi") == 0 ){

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
		string msg = "There was a problem opening the output file : " + filename;
		abortWithErrorMessage(msg);
	}

	for(unsigned int i=0; i<howMany;i++){
		inputFileStream >> result(i);
	}

	inputFileStream.close();

	return result;
}

//void ObjectiveFunction::readOnlyFunctionalValue(Design &d) const {
//	unsigned int howManyEntriesToRead = 1;
//	rowvec functionalValue(howManyEntriesToRead);
//
//
//	if(isHiFiEvaluation()){
//		functionalValue = readOutput(definition.outputFilename,howManyEntriesToRead);
//		d.trueValue = functionalValue(0);
//	}
//
//	if(isLowFiEvaluation()){
//
//		functionalValue = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
//		d.trueValueLowFidelity = functionalValue(0);
//	}
//
//
//}
//
//
//
//void ObjectiveFunction::readFunctionalValueAndTangent(Design &d) const {
//	unsigned int howManyEntriesToRead = 2;
//	rowvec resultBuffer(howManyEntriesToRead);
//
//
//	if(isHiFiEvaluation()){
//		resultBuffer = readOutput(definition.outputFilename,howManyEntriesToRead);
//		d.trueValue = resultBuffer(0);
//		d.tangentValue = resultBuffer(1);
//	}
//
//	if(isLowFiEvaluation()){
//		resultBuffer = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
//		d.trueValueLowFidelity = resultBuffer(0);
//		d.tangentValueLowFidelity = resultBuffer(1);
//	}
//
//}
//
//void ObjectiveFunction::readFunctionalValueAndAdjoint(Design &d) const {
//
//	assert(dim >0);
//
//	unsigned int howManyEntriesToRead = 1 + dim;
//	rowvec resultBuffer(howManyEntriesToRead);
//
//	rowvec gradient(dim, fill::zeros);
//	unsigned int offset = 1;
//
//	if(isHiFiEvaluation()){
//
//		resultBuffer = readOutput(definition.outputFilename,howManyEntriesToRead);
//		for (unsigned int i = 0; i < dim; i++) {
//			gradient(i) = resultBuffer(i + offset);
//		}
//		d.trueValue = resultBuffer(0);
//		d.gradient = gradient;
//
//	}
//	if(isLowFiEvaluation()){
//		resultBuffer = readOutput(definition.outputFilenameLowFi,howManyEntriesToRead);
//		for (unsigned int i = 0; i < dim; i++) {
//			gradient(i) = resultBuffer(i + offset);
//		}
//		d.trueValueLowFidelity = resultBuffer(0);
//		d.gradientLowFidelity = gradient;
//
//	}
//
//}
//
//void ObjectiveFunction::readOutputDesign(Design &d) const{
//
//	if(evaluationMode.compare("primal") == 0 || evaluationMode.compare("primalLowFi") == 0){
//
//		readOnlyFunctionalValue(d);
//	}
//
//	if(evaluationMode.compare("tangent") == 0 || evaluationMode.compare("tangentLowFi") == 0 ){
//
//		readFunctionalValueAndTangent(d);
//	}
//
//	if(evaluationMode.compare("adjoint") == 0 || evaluationMode.compare("adjointLowFi") == 0){
//
//		readFunctionalValueAndAdjoint(d);
//	}
//
//
//}

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

	setEvaluationMode("primal");
	writeDesignVariablesToFile(d);
	evaluateObjectiveFunction();

	rowvec result = readOutput(definition.outputFilename, 1);

	d.trueValue = result(0);

}


void ObjectiveFunction::evaluateDesignGradient(Design &d){

	assert(dim>0);
	assert(d.designParameters.size() == dim);

	setEvaluationMode("adjoint");

	writeDesignVariablesToFile(d);
	evaluateGradient();

	rowvec result = readOutput(definition.outputGradientFilename, dim);
	d.gradient = result;

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



void ObjectiveFunction::evaluateGradient(void) const{



	std::string runCommand;
	runCommand.clear();

	if(isHiFiEvaluation()){

		assert(isNotEmpty(definition.executableNameGradient));
		runCommand = getExecutionCommand(definition.executableNameGradient);


	}
	if(isLowFiEvaluation()){

		assert(isNotEmpty(definition.executableNameLowFiGradient));
		runCommand = getExecutionCommand(definition.executableNameLowFi);

	}

	if(isNotEmpty(runCommand)){

		output.printMessage("Calling executable for the objective function:", definition.name);

		int systemReturn = system(runCommand.c_str()) ;

		if(systemReturn == -1){

			string msg = "A process for the objective function/constraint execution could not be created, or its status could not be retrieved";

			abortWithErrorMessage(msg);

		}

		else{


			printWaitStatusIfSystemCallFails(systemReturn);
		}

	}

}



void ObjectiveFunction::evaluateObjectiveFunction(void){

	assert(isNotEmpty(definition.designVectorFilename));

	std::string runCommand;
	runCommand.clear();

	if(isHiFiEvaluation()){


		if(isNotEmpty(definition.executableName)){
			runCommand = getExecutionCommand(definition.executableName);
		}


	}
	if(isLowFiEvaluation()){

		if(isNotEmpty(definition.executableNameLowFi)){
			runCommand = getExecutionCommand(definition.executableNameLowFi);
		}

	}

	if(isNotEmpty(runCommand)){

		output.printMessage("Calling executable for the objective function:", definition.name);

		int systemReturn = system(runCommand.c_str()) ;

		if(systemReturn == -1){

			string msg = "A process for the objective function/constraint execution could not be created, or its status could not be retrieved";

			abortWithErrorMessage(msg);

		}

		else{


			printWaitStatusIfSystemCallFails(systemReturn);
		}

	}

}

void ObjectiveFunction::printWaitStatusIfSystemCallFails(int status) const{

	if (WIFEXITED(status)) {

		int statusCode = WEXITSTATUS(status);
		if (statusCode == 0) {

			output.printMessage("Objective function/constraint execution is done!");
		} else {
			string msg  = "There has been some problem with the objective function/constraint execution!";
			abortWithErrorMessage(msg);

		}
	}
}



double ObjectiveFunction::interpolate(rowvec x) const{
	return surrogate->interpolate(x);
}

double ObjectiveFunction::interpolateUsingDerivatives(rowvec x) const{
	return surrogate->interpolateUsingDerivatives(x);
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


void ObjectiveFunction::removeVeryCloseSamples(const Design& globalOptimalDesign){


	surrogate->removeVeryCloseSamples(globalOptimalDesign);
	surrogate->updateModelWithNewData();

}

void ObjectiveFunction::removeVeryCloseSamples(const Design& globalOptimalDesign, std::vector<rowvec> samples){


	surrogate->removeVeryCloseSamples(globalOptimalDesign,samples);
	surrogate->updateModelWithNewData();

}


void ObjectiveFunction::setSigmaFactor(double factor){

	assert(factor>0.0);
	sigmaFactor = factor;

}

void ObjectiveFunction::setGlobalOptimalDesign(Design d){
	assert(ifSurrogateModelIsDefined);
	surrogate->setGlobalOptimalDesign(d);

}

