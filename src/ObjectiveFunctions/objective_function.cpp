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
#include <algorithm>  // Include this for std::find
#include <stdexcept>  // For std::invalid_argument
#include <vector>
#include <boost/process.hpp>

#include "./INCLUDE/objective_function.hpp"
#include "./INCLUDE/objective_function_logger.hpp"
#include "../Bounds/INCLUDE/bounds.hpp"

namespace Rodop{

ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){}

bool ObjectiveFunctionDefinition::checkIfDefinitionIsOk(void) const {

	// Check if the objective function is defined
	if (!ifDefined) {
		std::cout << "Error: Objective function definition flag (ifDefined) is false." << std::endl;
		return false;
	}

	// Check if the name is empty
	if (name.empty()) {
		std::cout << "Error: Objective function name is empty!" << std::endl;
		return false;
	}

	// Check if the design vector filename is empty
	if (designVectorFilename.empty()) {
		std::cout << "Error: Design vector filename is empty!" << std::endl;
		return false;
	}

	// Check if the output filename is empty
	if (outputFilename.empty()) {
		std::cout << "Error: Output filename is empty!" << std::endl;
		return false;
	}

	// Check if the high-fidelity training data name is empty
	if (nameHighFidelityTrainingData.empty()) {
		std::cout << "Error: High-fidelity training data name is empty!" << std::endl;
		return false;
	}

	// If multi-level optimization is enabled, check the required fields
	if (ifMultiLevel) {
		if (nameLowFidelityTrainingData.empty()) {
			std::cout << "Error: Low-fidelity training data name is empty!" << std::endl;
			return false;
		}

		if (outputFilenameLowFi.empty()) {
			std::cout << "Error: Low-fidelity output filename is empty!" << std::endl;
			return false;
		}

		// Ensure the low-fidelity and high-fidelity training data names are not the same
		if (nameLowFidelityTrainingData == nameHighFidelityTrainingData) {
			std::cout << "Error: Low-fidelity and high-fidelity training data names must be different!" << std::endl;
			return false;
		}
	}

	// If all checks pass, return true
	return true;
}

string ObjectiveFunctionDefinition::getNameOfSurrogateModel(SURROGATE_MODEL modelType) const {

	string modelName;
	if (modelType == ORDINARY_KRIGING)
		modelName = "Uses only functional values";

	if (modelType == GRADIENT_ENHANCED)
		modelName = "Uses gradients";

	if (modelType == TANGENT_ENHANCED)
		modelName = "Uses directional derivatives";

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

void ObjectiveFunctionDefinition::print(void) const {
	// Print basic information
	std::cout << "Name = " << name << "\n";

	// Check if user-defined function (UDF) is used
	if (!doesUseUDF) {
		std::cout << "Design vector filename = " << designVectorFilename << "\n";
		std::cout << "Training data = " << nameHighFidelityTrainingData << "\n";
		std::cout << "Output filename = " << outputFilename << "\n";
		std::cout << "Executable = " << executableName << "\n";
	} else {
		std::cout << "UDF = YES\n";
	}

	// If gradient executable is defined
	if (!executableNameGradient.empty()) {
		std::cout << "Executable for gradient = " << executableNameGradient << "\n";
		std::cout << "Output file name for gradient = " << outputGradientFilename << "\n";
	}

	// Print high-fidelity model
	printHighFidelityModel();

	// Multilevel check
	std::cout << "Multilevel = " << (ifMultiLevel ? "YES" : "NO") << "\n";

	// If multilevel optimization is enabled
	if (ifMultiLevel) {
		std::cout << "Low fidelity model:\n";
		std::cout << "\tTraining data = " << nameLowFidelityTrainingData << "\n";
		std::cout << "\tOutput filename = " << outputFilenameLowFi << "\n";

		// If gradient output filename is defined for low-fidelity model
		if (!outputFilenameLowFiGradient.empty()) {
			std::cout << "\tOutput filename for gradient = " << outputFilenameLowFiGradient << "\n";
		}

		std::cout << "\tExecutable = " << executableNameLowFi << "\n";

		// If gradient executable is defined for low-fidelity model
		if (!executableNameLowFiGradient.empty()) {
			std::cout << "\tExecutable for gradient = " << executableNameLowFiGradient << "\n";
		}

		// Print low-fidelity model
		printLowFidelityModel();
	}
}

std::string ObjectiveFunctionDefinition::toStringLowFidelityModel() const {
	std::ostringstream oss;
	oss << "\tSurrogate model = " << getNameOfSurrogateModel(modelLowFi) << "\n";
	return oss.str();
}


std::string ObjectiveFunctionDefinition::toString() const {
	std::ostringstream oss;

	// Basic information
	oss << "Name = " << name << "\n";

	// User-defined function (UDF) information
	if (!doesUseUDF) {
		oss << "Design vector filename = " << designVectorFilename << "\n";
		oss << "Training data = " << nameHighFidelityTrainingData << "\n";
		oss << "Output filename = " << outputFilename << "\n";
		oss << "Executable = " << executableName << "\n";
	} else {
		oss << "UDF = YES\n";
	}

	// Gradient executable details if available
	if (!executableNameGradient.empty()) {
		oss << "Uses gradient enhanced model" << "\n";
		oss << "Executable for gradient = " << executableNameGradient << "\n";
		oss << "Output file name for gradient = " << outputGradientFilename << "\n";
	}
	else{
		oss << "No gradient enhancement is active" << "\n";

	}

	// High-fidelity model
	// Call to printHighFidelityModel() would be here if it had a string variant

	// Check for multilevel and print related information
	oss << "Multilevel = " << (ifMultiLevel ? "YES" : "NO") << "\n";
	if (ifMultiLevel) {
		oss << "Low fidelity model:\n";
		oss << "\tTraining data = " << nameLowFidelityTrainingData << "\n";
		oss << "\tOutput filename = " << outputFilenameLowFi << "\n";

		// Low-fidelity gradient output filename if available
		if (!outputFilenameLowFiGradient.empty()) {
			oss << "\tOutput filename for gradient = " << outputFilenameLowFiGradient << "\n";
		}

		oss << "\tExecutable = " << executableNameLowFi << "\n";

		// Low-fidelity gradient executable if available
		if (!executableNameLowFiGradient.empty()) {
			oss << "\tExecutable for gradient = " << executableNameLowFiGradient << "\n";
		}

		// Low-fidelity model details
		oss << toStringLowFidelityModel();
	}

	return oss.str();
}


/**********************************************************************************************/


ObjectiveFunction::ObjectiveFunction(){}

void ObjectiveFunction::setEvaluationMode(const std::string& mode) {

	// You could also handle this more gracefully with an exception
	if (mode.empty()) {
		throw std::invalid_argument("Evaluation mode cannot be empty.");
	}
	validateEvaluationMode(mode);
	evaluationMode = mode;
}


void ObjectiveFunction::validateEvaluationMode(const std::string& mode) {
	static const std::vector<std::string> validModes = {"primal", "adjoint", "tangent"};
	if (std::find(validModes.begin(), validModes.end(), mode) == validModes.end()) {
		throw std::invalid_argument("Invalid evaluation mode: " + mode);
	}
}

void ObjectiveFunction::validateDataAddMode(const std::string& mode) {
	static const std::vector<std::string> validModes = {"primal", "adjoint", "tangent"};
	if (std::find(validModes.begin(), validModes.end(), mode) == validModes.end()) {
		throw std::invalid_argument("Invalid evaluation mode: " + mode);
	}
}


void ObjectiveFunction::setDataAddMode(const std::string& mode) {

	if (mode.empty()) {
		throw std::invalid_argument("Data add mode cannot be empty.");
	}
	validateDataAddMode(mode);
	// Set the data add mode
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

	ObjectiveFunctionLogger::getInstance().log(INFO,"Binding the surrogate model with the ORDINARY_KRIGING modeĺ...");
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithUniversalKrigingModel() {

	ObjectiveFunctionLogger::getInstance().log(INFO,"Binding the surrogate model with the UNIVERSAL_KRIGING modeĺ...");
	surrogateModel.setLinearRegressionOn();
	surrogate = &surrogateModel;
}



void ObjectiveFunction::bindWithMultiFidelityModel() {

	surrogateModelML.setIDHiFiModel(definition.modelHiFi);
	surrogateModelML.setIDLowFiModel(definition.modelLowFi);

	surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
	surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);


	/* TODO modify this ugly code */
	surrogateModelML.bindModels();

	ObjectiveFunctionLogger::getInstance().log(INFO,"Binding the surrogate model with the Multi-fidelity model...");
	surrogate = &surrogateModelML;

}

void ObjectiveFunction::bindSurrogateModelSingleFidelity() {
	if (definition.modelHiFi == ORDINARY_KRIGING) {
		bindWithOrdinaryKrigingModel();
	}
	if (definition.modelHiFi == UNIVERSAL_KRIGING) {
		bindWithUniversalKrigingModel();
	}

}

void ObjectiveFunction::bindSurrogateModel(void){

	if (!ifDefinitionIsSet) {
		throw std::runtime_error("Cannot bind with a surrogate model without proper objective function definition.");
	}


	if(definition.ifMultiLevel){
		bindWithMultiFidelityModel();
	}
	else{
		bindSurrogateModelSingleFidelity();
	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){

	definition = def;
	ifDefinitionIsSet = true;

}

void ObjectiveFunction::setNumberOfTrainingIterationsForSurrogateModel(unsigned int nIter){
	numberOfIterationsForSurrogateTraining = nIter;
}
void ObjectiveFunction::setFileNameReadInput(std::string fileName){
	if (fileName.empty()) {
		throw std::invalid_argument("ObjectiveFunction::setFileNameReadInput: File name cannot be empty.");
	}

	definition.outputFilename = fileName;
}
void ObjectiveFunction::setFileNameReadInputLowFidelity(std::string fileName){

	if (fileName.empty()) {
		throw std::invalid_argument("ObjectiveFunction::setFileNameReadInputLowFidelity: File name cannot be empty.");
	}
	definition.outputFilenameLowFi = fileName;
}

void ObjectiveFunction::setFileNameDesignVector(std::string fileName){
	assert(!fileName.empty());
	definition.designVectorFilename = fileName;
}

std::string ObjectiveFunction::getFileNameDesignVector(void) const{
	return definition.designVectorFilename;
}


void ObjectiveFunction::setFileNameTrainingData(std::string fileName){

	assert(!fileName.empty());
	definition.nameHighFidelityTrainingData = fileName;
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


void ObjectiveFunction::setParameterBounds(Bounds bounds) {
	// Ensure that if the dimension is greater than 0, it matches the bounds' dimension
	if (dim > 0) {
		if (dim != bounds.getDimension()) {
			throw std::invalid_argument("Mismatch between objective function dimension and bounds dimension.");
		}
	}

	// Assert that the bounds are properly set
	if (!bounds.areBoundsSet()) {
		throw std::invalid_argument("Bounds have not been set properly.");
	}

	// Set the bounds and flag that they are set
	boxConstraints = bounds;
	ifParameterBoundsAreSet = true;
}

MultiLevelModel ObjectiveFunction::getSurrogateModelML(void) const{
	return surrogateModelML;
}



void ObjectiveFunction::initializeSurrogate() {
	// Check if the parameter bounds, definition, and dimensions are set
	if (!ifParameterBoundsAreSet) {
		throw std::runtime_error("Parameter bounds must be set before initializing the surrogate.");
	}

	if (!ifDefinitionIsSet) {
		throw std::runtime_error("Objective function definition must be set before initializing the surrogate.");
	}

	if (dim <= 0) {
		throw std::runtime_error("The dimension of the problem must be greater than zero.");
	}

	// Bind the surrogate model
	bindSurrogateModel();

	// Set surrogate properties
	surrogate->setName(definition.name);
	surrogate->setDimension(dim);
	surrogate->setBoxConstraints(boxConstraints);
	surrogate->setNameOfInputFile(definition.nameHighFidelityTrainingData);

	// Log the initialization process
	std::string msg = "Reading training data for surrogate model: " + definition.nameHighFidelityTrainingData;
	ObjectiveFunctionLogger::getInstance().log(INFO,msg);

	// Try to read data and handle any file reading errors
	try {
		surrogate->readData();
	} catch (const std::exception& e) {
		throw std::runtime_error("Failed to read training data: " + std::string(e.what()));
	}

	// Normalize the data
	surrogate->normalizeData();

	// Initialize the surrogate model
	surrogate->initializeSurrogateModel();

	// Set the number of iterations for training
	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

	// Mark the surrogate model as initialized
	ifInitialized = true;

	// Log success
	msg = "Surrogate model successfully initialized.";
	ObjectiveFunctionLogger::getInstance().log(INFO,msg);

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

void ObjectiveFunction::calculateExpectedImprovementUsingDerivatives(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);
	double sigma = sqrt(ssqr);

	ftilde = surrogate->interpolateUsingDerivatives(designCalculated.dv);

	designCalculated.sigma = sigma;

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

	designCalculated.valueAcquisitionFunction = expectedImprovementValue;
	designCalculated.objectiveFunctionValue = ftilde;
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


	designCalculated.valueAcquisitionFunction = log(expectedImprovementValue+1e-14);
	designCalculated.objectiveFunctionValue = ftilde;

}


void ObjectiveFunction::calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double	sigma = sqrt(ssqr)	;
	designCalculated.sigma = sigma;

	double PI = designCalculated.calculateProbalityThatTheEstimateIsLessThanAValue(sampleMinimum);

	designCalculated.valueAcquisitionFunction = PI;

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


std::string ObjectiveFunction::getExecutionCommand(const std::string& exename) const {
	std::string command;

	// Check if the input ends with ".py", indicating a Python script
	if (exename.size() >= 3 && exename.substr(exename.size() - 3) == ".py") {
		command = "python " + exename;  // Prepends 'python ' to the command
	} else {
		// Check if the exename already contains a path (i.e., contains '/')
		if (exename.find('/') != std::string::npos) {
			command = exename;  // Use the full path as the command
		} else {
			command = "./" + exename;  // Prepend './' for relative path executables
		}
	}

	return command;
}


void ObjectiveFunction::addDesignToData(Design &d){

	assert(!definition.nameHighFidelityTrainingData.empty());
	assert(ifInitialized);
	assert(!addDataMode.empty());


	if(definition.ifMultiLevel == false){

		vec newsample;

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


		assert(newsample.getSize()>0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		vec newsampleHiFi;
		vec newsampleLowFi;

		if(addDataMode.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.getSize() >0);
			assert(newsampleHiFi.getSize()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(addDataMode.compare("primalHiFiAdjointLowFi") == 0 ){

			newsampleHiFi  = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();

			assert(newsampleLowFi.getSize() >0);
			assert(newsampleHiFi.getSize()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}




	}

}


void ObjectiveFunction::addDesignToData(Design &d, string how){

	if(definition.nameHighFidelityTrainingData.empty()){
		std::string msg = "File name for the training data is empty. ";
		ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
		throw std::runtime_error(msg);
	}

	if(!ifInitialized){
		std::string msg = "Objective function is not initialized. ";
		ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
		throw std::runtime_error(msg);

	}

	if(how.empty()){

		std::string msg = "Data add mode is not specified.";
		ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
		throw std::runtime_error(msg);
	}

	if(definition.ifMultiLevel == false){

		vec newsample;

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


		assert(newsample.getSize() > 0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		vec newsampleHiFi;
		vec newsampleLowFi;

		if(how.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.getSize() >0);
			assert(newsampleHiFi.getSize()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(how.compare("primalHiFiAdjointLowFi") == 0 ){

			newsampleHiFi  = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();

			assert(newsampleLowFi.getSize() >0);
			assert(newsampleHiFi.getSize()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}




	}

}



void ObjectiveFunction::addLowFidelityDesignToData(Design &d){

	assert(!definition.nameLowFidelityTrainingData.empty());
	assert(ifInitialized);
	assert(!addDataMode.empty());
	assert(definition.ifMultiLevel == true);

	vec newsampleLowFi;

	if(addDataMode.compare("primalLowFidelity") == 0 ){
		newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();
		assert(newsampleLowFi.getSize()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);

	}

	if(addDataMode.compare("adjointLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();
		assert(newsampleLowFi.getSize()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}

	if(addDataMode.compare("tangentLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithTangentLowFi();
		assert(newsampleLowFi.getSize()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}





}


vec ObjectiveFunction::readOutput(const std::string &filename, unsigned int howMany) const {

	// Ensure that the filename is not empty
	if (filename.empty()) {
		throw std::invalid_argument("Filename is empty.");
	}

	vec result(howMany);

	// Open the file stream
	std::ifstream inputFileStream(filename);
	if (!inputFileStream.is_open()) {
		std::string msg = "There was a problem opening the output file: " + filename;
		ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
		throw std::runtime_error(msg);
	}

	// Read the output values from the file
	for (unsigned int i = 0; i < howMany; i++) {
		if (!(inputFileStream >> result(i))) {
			std::string msg = "Failed to read value " + std::to_string(i) + " from file: " + filename;
			ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
			throw std::runtime_error(msg);
		}
	}

	inputFileStream.close();
	return result;
}



void ObjectiveFunction::writeDesignVariablesToFile(Design &d) const{

	assert(d.designParameters.getSize() == dim);
	assert(!definition.designVectorFilename.empty());

	std::ofstream outputFileStream(definition.designVectorFilename);

	if (!outputFileStream.is_open()) {
		abort();
	}

	for(unsigned int i=0; i<dim; i++) {

		outputFileStream << d.designParameters(i) << std::endl;
	}

	if(evaluationMode.compare("tangent") == 0 || evaluationMode.compare("tangentLowFi") == 0){

		assert(d.tangentDirection.getSize() == dim);
		for(unsigned int i=0; i<dim; i++) {

			outputFileStream << d.tangentDirection(i) << std::endl;
		}

	}
	outputFileStream.close();

}

void ObjectiveFunction::evaluateDesign(Design &d) {
	// Ensure the design parameters match the dimensionality of the problem
	if (d.designParameters.getSize() != dim) {
		throw std::invalid_argument("Design parameter size does not match the problem dimension.");
	}

	// Set evaluation mode to "primal"
	setEvaluationMode("primal");

	// If the objective function pointer does not exist, perform file-based evaluation
	if (!doesObjectiveFunctionPtrExist) {
		// Write design variables to a file
		writeDesignVariablesToFile(d);

		// Log the evaluation process

		printInfoToLog("Evaluating objective function via external execution.");

		// Evaluate the objective function externally
		evaluateObjectiveFunction();

		// Read the result from the output file
		vec result = readOutput(definition.outputFilename, 1);

		// Ensure the result is valid (for example, at least one value)
		if (result.getSize() == 0) {
			throw std::runtime_error("Failed to read output from the external objective function evaluation.");
		}

		double objectiveFunction = result(0);

		ObjectiveFunctionLogger::getInstance().log(INFO, "Objective function value = " + std::to_string(objectiveFunction));

		// Set the true value of the design based on the result
		d.trueValue = result(0);
	}
	// If the objective function pointer exists, evaluate directly
	else {
		// Log direct evaluation
		ObjectiveFunctionLogger::getInstance().log(INFO, "Evaluating objective function directly.");
		// Evaluate the objective function directly using the design parameters
		double objectiveFunction = evaluateObjectiveFunctionDirectly(d.designParameters);

		ObjectiveFunctionLogger::getInstance().log(INFO, "Objective function value = " + std::to_string(objectiveFunction));

		// Set the true value of the design based on the result
		d.trueValue = objectiveFunction;
	}
}

void ObjectiveFunction::evaluateDesignGradient(Design &d) {
	// Ensure the dimension is valid
	if (dim <= 0) {
		throw std::runtime_error("The dimension of the problem must be greater than zero.");
	}

	// Ensure the design parameters match the problem dimension
	if (d.designParameters.getSize() != dim) {
		throw std::invalid_argument("Design parameter size does not match the problem dimension.");
	}

	// Set evaluation mode to "adjoint" for gradient calculation
	setEvaluationMode("adjoint");

	// Write design variables to a file
	writeDesignVariablesToFile(d);

	// Log the gradient evaluation process
	ObjectiveFunctionLogger::getInstance().log(INFO, "Evaluating the gradient of the objective function.");

	// Evaluate the gradient using an external process
	evaluateGradient();

	// Read the gradient result from the output file
	vec result = readOutput(definition.outputGradientFilename, dim);

	// Ensure the result is valid and matches the problem dimension
	if (result.getSize() != dim) {
		throw std::runtime_error("The size of the gradient result does not match the problem dimension.");
	}

	ObjectiveFunctionLogger::getInstance().log(INFO, "Gradient = " + result.toString() );


	// Assign the result to the design's gradient
	d.gradient = result;

	// Log success
	ObjectiveFunctionLogger::getInstance().log(INFO, "Gradient evaluation completed successfully.");
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

void ObjectiveFunction::evaluateGradient() const {
	using namespace boost::process;

	if (evaluationMode.empty()) {
		string msg =
				"ObjectiveFunction::evaluateGradient: evaluation mode is not set";
		printErrorToLog(msg);
		throw std::runtime_error(msg);
	}
	if (evaluationMode != "adjoint" && evaluationMode != "adjointLowFi") {
		string msg =
				"ObjectiveFunction::evaluateGradient: evaluation mode is invalid for this method";
		printErrorToLog(msg);
		throw std::runtime_error(msg);
	}

	std::string runCommand;
	// Determine if it's a HiFi or LowFi evaluation and construct the appropriate command
	if (isHiFiEvaluation()) {
		if (definition.executableNameGradient.empty()) {
			throw std::runtime_error("High-fidelity gradient executable name is not set.");
		}
		runCommand = getExecutionCommand(definition.executableNameGradient);
	} else if (isLowFiEvaluation()) {
		if (definition.executableNameLowFiGradient.empty()) {
			throw std::runtime_error("Low-fidelity gradient executable name is not set.");
		}
		runCommand = getExecutionCommand(definition.executableNameLowFiGradient);
	}

	if (!runCommand.empty()) {
		try {

			printInfoToLog("Executing command = " + runCommand);
			// Execute the command and wait for completion
			child c(runCommand);
			c.wait(); // Wait for the process to finish

			if (c.exit_code() != 0) {
				printErrorToLog("ObjectiveFunction::evaluateGradient: exit_code = " + std::to_string(c.exit_code()));
				throw std::runtime_error("Gradient vector evaluation failed with exit code: " + std::to_string(c.exit_code()));
			}
		} catch (const std::exception& e) {
			std::string msg = std::string("Error during gradient vector evaluation: ") + e.what();
			throw std::runtime_error(msg);
		}
	} else {
		printErrorToLog("No valid command found to evaluate the gradient vector.");
		printErrorToLog("Executable = " + definition.executableNameGradient);
		printErrorToLog("Executable Low Fi= " + definition.executableNameLowFiGradient);
		throw std::runtime_error("No valid command found to evaluate the gradient.");
	}
}



void ObjectiveFunction::printInfoToLog(const string &msg) const{
	ObjectiveFunctionLogger::getInstance().log(INFO, msg);
}

void ObjectiveFunction::printErrorToLog(const string &msg) const{
	ObjectiveFunctionLogger::getInstance().log(ERROR, msg);
}
void ObjectiveFunction::printWarningToLog(const string &msg) const{
	ObjectiveFunctionLogger::getInstance().log(WARNING, msg);
}

double ObjectiveFunction::evaluateObjectiveFunctionDirectly(const vec &x) {
	// Ensure the objective function pointer is set
	if (objectiveFunctionPtr == nullptr) {
		throw std::runtime_error("Objective function pointer is not set.");
	}

	// Log the evaluation process
	printInfoToLog("Evaluating objective function directly with provided vector.");

	// Call the objective function pointer with the input vector
	return objectiveFunctionPtr(x.getPointer());
}

void ObjectiveFunction::checkEvaluationModeForPrimalExecution() const {
	if (evaluationMode.empty()) {
		string msg =
				"ObjectiveFunction::evaluateObjectiveFunction: evaluation mode is not set";
		printErrorToLog(msg);
		throw std::runtime_error(msg);
	}
	if (evaluationMode != "primal" && evaluationMode != "primalLowFi") {
		string msg =
				"ObjectiveFunction::evaluateObjectiveFunction: evaluation mode is invalid for this method";
		printErrorToLog(msg);
		throw std::runtime_error(msg);
	}
}

void ObjectiveFunction::evaluateObjectiveFunction() const {
	using namespace boost::process;

	checkEvaluationModeForPrimalExecution();

	std::string runCommand;
	if (isHiFiEvaluation() && !definition.executableName.empty()) {
		runCommand = getExecutionCommand(definition.executableName);
	} else if (isLowFiEvaluation() && !definition.executableNameLowFi.empty()) {
		runCommand = getExecutionCommand(definition.executableNameLowFi);
	}

	if (!runCommand.empty()) {
		try {
			// Execute the command and wait for completion
			child c(runCommand);
			c.wait(); // Wait for the process to finish

			if (c.exit_code() != 0) {
				printErrorToLog("ObjectiveFunction::evaluateObjectiveFunction: exit_code = " + std::to_string(c.exit_code()));
				throw std::runtime_error("Objective function execution failed with exit code: " + std::to_string(c.exit_code()));
			}
		} catch (const std::exception& e) {
			std::string msg = std::string("Error during objective function execution: ") + e.what();
			throw std::runtime_error(msg);
		}
	} else {
		printWarningToLog("No valid command found to execute the objective function.");
	}
}



double ObjectiveFunction::interpolate(vec x) const{
	return surrogate->interpolate(x);
}

double ObjectiveFunction::interpolateUsingDerivatives(vec x) const{
	return surrogate->interpolateUsingDerivatives(x);
}

pair<double, double> ObjectiveFunction::interpolateWithVariance(vec x) const{

	double ftilde,sigmaSqr;
	surrogate->interpolateWithVariance(x, &ftilde, &sigmaSqr);

	pair<double, double> result;
	result.first = ftilde;
	result.second = sqrt(sigmaSqr);

	return result;
}


void ObjectiveFunction::print(void) const{
	std::cout<<"\n================ Objective function definition =========================\n";
	definition.print();
	std::cout<< "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
	std::cout<<"==========================================================================\n";
}


std::string ObjectiveFunction::toString() const {
	std::ostringstream oss;
	oss << "\n================ Objective function definition =========================\n";
	oss << definition.toString();  // Assuming ObjectiveFunctionDefinition has a toString method
	oss << "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
	oss << "==========================================================================\n";
	return oss.str();
}

std::string  ObjectiveFunction::generateFormattedString(std::string msg, char c, int totalLength) const{
	if (totalLength < 0) {
		throw std::invalid_argument("Number of characters must be non-negative.");
	}

	if(msg.length()%2 == 1){
		msg+=" ";
	}

	int numEquals = static_cast<int>((totalLength - msg.length() - 2)/2);


	std::string border(numEquals, c);



	std::ostringstream oss;
	oss << border << " " << msg << " " << border;

	return oss.str();
}



string ObjectiveFunction::generateOutputString(void) const{


	std::string outputMsg;
	string tag = "Objective function definition";
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

void ObjectiveFunction::printSurrogate(void) const{
	surrogate->printSurrogateModel();
}

void ObjectiveFunction::setSigmaFactor(double factor){

	assert(factor>0.0);
	sigmaFactor = factor;

}


void ObjectiveFunction::setFunctionPtr(ObjectiveFunctionPtr func) {
	if (func == nullptr) {
		throw std::invalid_argument("Function pointer is null");
	}

	objectiveFunctionPtr = func;
	doesObjectiveFunctionPtrExist = true;
}




/** Returns the pdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_pdf(x) with mu and sigma
 *
 */


double ObjectiveFunction::pdf(double x, double m, double s) const
{
	double a = (x - m) / s;

	return INVSQRT2PI / s * std::exp(-0.5 * a * a);
}



/** Returns the cdf of x, given the distribution described by mu and sigma..
 *
 *  CFD(x0) = Pr(x < x0)
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_cdf(x) with mu and sigma
 *
 */
double ObjectiveFunction::cdf(double x0, double mu, double sigma) const
{

	double inp = (x0 - mu) / (sigma * SQRT2);
	double result = 0.5 * (1.0 + erf(inp));
	return result;
}





} /*Namespace Rodop */

