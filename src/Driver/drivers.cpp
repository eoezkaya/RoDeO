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
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <cassert>


#include "../Optimizers/INCLUDE/optimization.hpp"
#include "../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"


#include "./INCLUDE/drivers.hpp"
#include "./INCLUDE/configkey.hpp"
#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

void RoDeODriver::addConfigKeysObjectiveFunction() {

	configKeysObjectiveFunction.add(ConfigKey("NAME", "string"));
	configKeysObjectiveFunction.add(ConfigKey("DESIGN_VECTOR_FILE", "string"));
	configKeysObjectiveFunction.add(ConfigKey("MULTILEVEL_MODEL", "string"));
	configKeysObjectiveFunction.add(ConfigKey("WARM_START", "string"));

	configKeysObjectiveFunction.add(ConfigKey("OUTPUT_FILE", "stringVector"));
	configKeysObjectiveFunction.add(ConfigKey("PATH", "stringVector"));
	configKeysObjectiveFunction.add(ConfigKey("EXECUTABLE", "stringVector"));
	configKeysObjectiveFunction.add(ConfigKey("SURROGATE_MODEL", "stringVector"));
	configKeysObjectiveFunction.add(ConfigKey("FILENAME_TRAINING_DATA", "stringVector"));

	configKeysObjectiveFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS", "int"));

}

void RoDeODriver::addConfigKeysConstraintFunctions() {

	configKeysConstraintFunction.add(ConfigKey("DEFINITION", "string"));
	configKeysConstraintFunction.add(ConfigKey("DESIGN_VECTOR_FILE", "string"));
	configKeysConstraintFunction.add(ConfigKey("MULTILEVEL_MODEL", "string"));
	configKeysConstraintFunction.add(ConfigKey("WARM_START", "string"));

	configKeysConstraintFunction.add(ConfigKey("OUTPUT_FILE", "stringVector"));
	configKeysConstraintFunction.add(ConfigKey("PATH", "stringVector"));
	configKeysConstraintFunction.add(ConfigKey("EXECUTABLE", "stringVector"));
	configKeysConstraintFunction.add(ConfigKey("SURROGATE_MODEL", "stringVector"));
	configKeysConstraintFunction.add(ConfigKey("FILENAME_TRAINING_DATA", "stringVector"));

	configKeysConstraintFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS", "int"));

}

void RoDeODriver::addConfigKeysSurrogateModelTest() {

	configKeys.add(ConfigKey("MULTILEVEL_MODEL", "string"));
	configKeys.add(ConfigKey("WARM_START", "string"));
	configKeys.add(ConfigKey("FILENAME_TEST_DATA", "string"));

	configKeys.add(ConfigKey("FILENAME_TRAINING_DATA", "stringVector"));
	configKeys.add(ConfigKey("SURROGATE_MODEL", "stringVector"));

	configKeys.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS", "int"));
}

void RoDeODriver::addConfigKeysOptimization() {

	configKeys.add(ConfigKey("VISUALIZATION", "string"));
	configKeys.add(ConfigKey("DISPLAY", "string"));
	configKeys.add(ConfigKey("ZOOM_IN", "string"));
	configKeys.add(ConfigKey("TARGET_VALUE_FOR_VARIABLE_SAMPLE_WEIGHTS", "string"));

	configKeys.add(ConfigKey("ZOOM_IN_CONTRACTION_FACTOR", "double"));

	configKeys.add(ConfigKey("DISCRETE_VARIABLES", "doubleVector"));
	configKeys.add(ConfigKey("DISCRETE_VARIABLES_VALUE_INCREMENTS", "doubleVector"));

	configKeys.add(ConfigKey("ZOOM_IN_HOW_OFTEN", "int"));
	configKeys.add(ConfigKey("MAX_NUMBER_OF_INNER_ITERATIONS", "int"));
	configKeys.add(ConfigKey("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS", "int"));

}

void RoDeODriver::addConfigKeysGeneral() {
	configKeys.add(ConfigKey("PROBLEM_TYPE", "string"));
	configKeys.add(ConfigKey("PROBLEM_NAME", "string"));
	configKeys.add(ConfigKey("DIMENSION", "int"));
	configKeys.add(ConfigKey("UPPER_BOUNDS", "doubleVector"));
	configKeys.add(ConfigKey("LOWER_BOUNDS", "doubleVector"));
}

void RoDeODriver::addAvailableSurrogateModels() {
	availableSurrogateModels.push_back("ORDINARY_KRIGING");
	availableSurrogateModels.push_back("UNIVERSAL_KRIGING");
	availableSurrogateModels.push_back("LINEAR_REGRESSION");
	availableSurrogateModels.push_back("TANGENT_ENHANCED");
	availableSurrogateModels.push_back("GRADIENT_ENHANCED");
}

RoDeODriver::RoDeODriver(){


	addConfigKeysObjectiveFunction();
	addConfigKeysConstraintFunctions();
	addConfigKeysGeneral();
	addConfigKeysSurrogateModelTest();
	addConfigKeysOptimization();
	addAvailableSurrogateModels();
}



void RoDeODriver::setDisplayOn(void){

	configKeys.assignKeywordValue("DISPLAY",std::string("ON"));
	output.ifScreenDisplay = true;

}
void RoDeODriver::setDisplayOff(void){

	configKeys.assignKeywordValue("DISPLAY",std::string("OFF"));
	output.ifScreenDisplay = false;
}


string RoDeODriver::removeComments(const string &configText) const{

	assert(isNotEmpty(configText));

	string result;
	istringstream buffer(configText);
	string line;
	while (getline(buffer, line)) {
		if(line[0] != '#' && line[0] != '%') {
			result.append(line + "\n");
		}
	}

	return result;

}
void RoDeODriver::readConfigFile(void){

	assert(isNotEmpty(configFileName));

	std::string msg = "Reading the configuration file: ";
	output.printMessage(msg, configFileName);

	std::string stringCompleteFile;

	readFileToaString(configFileName, stringCompleteFile);

	string configFileWithoutComments = removeComments(stringCompleteFile);

	msg = "Configuration file = ";
	output.printMessage(msg);
	output.printMessage(configFileWithoutComments);

	extractConfigDefinitionsFromString(configFileWithoutComments);

	extractObjectiveFunctionDefinitionFromString(configFileWithoutComments);

	extractConstraintDefinitionsFromString(configFileWithoutComments);

	checkConsistencyOfConfigParams();

	msg = "Parsing of the configuration file is done...";
	output.printMessage(msg);
}


void RoDeODriver::setConfigFilename(std::string filename){

	configFileName = filename;


}

void RoDeODriver::checkIfObjectiveFunctionNameIsDefined(void) const{

	ConfigKey configKeyName = configKeysObjectiveFunction.getConfigKey("NAME");
	configKeyName.abortIfNotSet();

}


void RoDeODriver::checkIfProblemDimensionIsSetProperly(void) const{

	configKeys.abortifConfigKeyIsNotSet("DIMENSION");

	int dimension = configKeys.getConfigKeyIntValue("DIMENSION");

	if(dimension > 1000){
		abortWithErrorMessage("Problem dimension is too large, did you set DIMENSION properly?");
	}
	if(dimension <= 0){
		abortWithErrorMessage("Problem dimension must be positive, did you set DIMENSION properly?");
	}

}


void RoDeODriver::checkIfBoxConstraintsAreSetPropertly(void) const{

	output.printMessage("Checking box constraint settings...");

	ConfigKey ub = configKeys.getConfigKey("UPPER_BOUNDS");
	ConfigKey lb = configKeys.getConfigKey("LOWER_BOUNDS");

	ConfigKey dimension = configKeys.getConfigKey("DIMENSION");
	int dim = dimension.intValue;

	ub.abortIfNotSet();
	lb.abortIfNotSet();

	int sizeUb = ub.vectorDoubleValue.size();
	int sizeLb = lb.vectorDoubleValue.size();

	if(sizeUb != sizeLb || sizeUb != dim){
		abortWithErrorMessage("Dimension of bounds does not match with the problem dimension, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?");
	}

	for(int i=0; i<dim; i++){

		if( ub.vectorDoubleValue(i) <= lb.vectorDoubleValue(i)){
			abortWithErrorMessage("Lower bounds cannot be greater or equal than upper bounds, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?");
		}

	}

}


void RoDeODriver::checkSettingsForSurrogateModelTest(void) const{


	checkIfSurrogateModelTypeIsOK();

	configKeys.abortifConfigKeyIsNotSet("FILENAME_TRAINING_DATA");
	configKeys.abortifConfigKeyIsNotSet("FILENAME_TEST_DATA");

}

void RoDeODriver::abortIfModelTypeIsInvalid(const string &modelName) const {
	bool ifTypeIsOK = isIntheList(availableSurrogateModels, modelName);
	if (!ifTypeIsOK) {
		std::cout
		<< "ERROR: Surrogate model is not available, did you set SURROGATE_MODEL properly?\n";
		std::cout << "Available surrogate models:\n";
		printVector(availableSurrogateModels);
		abort();
	}
}

void RoDeODriver::checkIfSurrogateModelTypeIsOkMultiFidelity() const {
	ConfigKey multilevelModel = configKeys.getConfigKey("MULTILEVEL_MODEL");
	std::string multilevel;
	multilevel = configKeys.getConfigKeyStringValue("MULTILEVEL_MODEL");
	if (checkIfOn(multilevel)) {
		string modelName = configKeys.getConfigKeyStringVectorValueAtIndex(
				"SURROGATE_MODEL", 1);
		abortIfModelTypeIsInvalid(modelName);
	}
}

void RoDeODriver::checkIfSurrogateModelTypeIsOK(void) const{

	ConfigKey surrogateModelType = configKeys.getConfigKey("SURROGATE_MODEL");

	surrogateModelType.abortIfNotSet();

	string modelName = configKeys.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",0);
	abortIfModelTypeIsInvalid(modelName);

	checkIfSurrogateModelTypeIsOkMultiFidelity();
}





void RoDeODriver::checkSettingsForOptimization(void) const{

	checkIfProblemDimensionIsSetProperly();
	checkIfBoxConstraintsAreSetPropertly();

	configKeys.abortifConfigKeyIsNotSet("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

}

void RoDeODriver::checkConsistencyOfConfigParams(void) const{

	output.printMessage("Checking consistency of the configuration parameters...");

	checkIfProblemTypeIsSetProperly();

	std::string type = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

	if(type == "SURROGATE_TEST"){
		checkSettingsForSurrogateModelTest();
	}

	if(type == "Optimization" || type == "OPTIMIZATION"){
		checkSettingsForOptimization();
	}

}

void RoDeODriver::checkIfProblemTypeIsSetProperly(void) const{


	ConfigKey problemType = configKeys.getConfigKey("PROBLEM_TYPE");
	problemType.abortIfNotSet();

	bool ifProblemTypeIsValid = checkifProblemTypeIsValid(problemType.stringValue);

	if(!ifProblemTypeIsValid){

		std::cout<<"ERROR: Problem type is not valid, did you set PROBLEM_TYPE properly?\n";
		std::cout<<"Valid problem types: OPTIMIZATION, SURROGATE_TEST\n";
		abort();

	}

}


bool RoDeODriver::checkifProblemTypeIsOptimization(std::string s) const{

	if(isEqual(s,"OPTIMIZATION")) return true;
	if(isEqual(s,"optimization")) return true;
	if(isEqual(s,"Optimization")) return true;

	return false;
}


bool RoDeODriver::checkifProblemTypeIsSurrogateTest(std::string s) const{

	if(isEqual(s,"SURROGATE_TEST")) return true;
	if(isEqual(s,"surrogate_test")) return true;
	if(isEqual(s,"Surrogate_Test")) return true;

	return false;
}

bool RoDeODriver::checkifProblemTypeIsValid(std::string s) const{

	if(checkifProblemTypeIsOptimization(s)) return true;
	if(checkifProblemTypeIsSurrogateTest(s)) return true;

	return false;
}




void RoDeODriver::checkConsistencyOfObjectiveFunctionDefinition(void) const{

	assert(definitionObjectiveFunction.ifDefined);

	string exeName = definitionObjectiveFunction.executableName;

	if(isEmpty(exeName)){
		abortWithErrorMessage("Objective function EXECUTABLE is undefined");
	}

	string name = definitionObjectiveFunction.name;

	if(isEmpty(exeName)){
		abortWithErrorMessage("Objective function NAME is undefined");
	}

	string filenameDesignVector = definitionObjectiveFunction.designVectorFilename;
	if(isEmpty(filenameDesignVector)){
		abortWithErrorMessage("Objective function DESIGN_VECTOR_FILE is undefined");
	}

	string outputFileName = definitionObjectiveFunction.outputFilename;

	if(isEmpty(outputFileName)){
		abortWithErrorMessage("Objective function OUTPUT_FILE is undefined");
	}

	string filenameTrainingData = definitionObjectiveFunction.nameHighFidelityTrainingData;

	if(isEmpty(filenameTrainingData)){
		abortWithErrorMessage("Objective function FILENAME_TRAINING_DATA is undefined");
	}


	bool ifMultiLevelIsActive = definitionObjectiveFunction.ifMultiLevel;

	if(ifMultiLevelIsActive){

		string exeNameLowFi = definitionObjectiveFunction.executableNameLowFi;

		if(isEmpty(exeNameLowFi)){
			abortWithErrorMessage("Objective function EXECUTABLE for the low-fidelity model is undefined");
		}
		string filenameTrainingDataLowFi = definitionObjectiveFunction.nameLowFidelityTrainingData;

		if(isEmpty(filenameTrainingDataLowFi)){
			abortWithErrorMessage("Objective function FILENAME_TRAINING_DATA for the low-fidelity model is undefined");
		}

		string outputFileNameLowFi = definitionObjectiveFunction.outputFilenameLowFi;

		if(isEmpty(outputFileNameLowFi)){
			abortWithErrorMessage("Objective function OUTPUT_FILE for the low-fidelity model is undefined");
		}
	}

}


ObjectiveFunction RoDeODriver::setObjectiveFunction(void) const{

	std::string objFunName = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");
	int dim = configKeys.getConfigKeyIntValue("DIMENSION");

	ObjectiveFunction objFunc;
	objFunc.setDimension(dim);

	checkConsistencyOfObjectiveFunctionDefinition();

	objFunc.setParametersByDefinition(definitionObjectiveFunction);

	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	Bounds boxConstraints(lb,ub);
	objFunc.setParameterBounds(boxConstraints);


	unsigned int nIterForSurrogateTraining = 10000;

	string key = "NUMBER_OF_TRAINING_ITERATIONS";
	if(configKeysObjectiveFunction.ifConfigKeyIsSet(key)){
		nIterForSurrogateTraining = configKeysObjectiveFunction.getConfigKeyIntValue(key);
	}

	objFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);


	string isWarmStartOn = configKeysObjectiveFunction.getConfigKeyStringValue("WARM_START");

	if(checkIfOn(isWarmStartOn)){
		objFunc.ifWarmStart	 = true;
	}



	return objFunc;


}


void RoDeODriver::printAllConstraintDefinitions(void) const{

	std::cout<<"\nList of all constraint definitions:\n";

	for ( auto i = constraintDefinitions.begin(); i != constraintDefinitions.end(); i++ ) {

		i->print();
	}


}

void RoDeODriver::printObjectiveFunctionDefinition(void) const{
	definitionObjectiveFunction.print();
}

void RoDeODriver::parseConstraintDefinitionMultiFidelity(ObjectiveFunctionDefinition constraintFunctionDefinition) {

	string multilevel = configKeysConstraintFunction.getConfigKeyStringValue("MULTILEVEL_MODEL");

	if (checkIfOn(multilevel)) {
		constraintFunctionDefinition.ifMultiLevel = true;
		std::string executableNameLowFi =
				configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE", 1);
		if (executableNameLowFi.empty()) {
			string msg ="EXECUTABLE for the low fidelity model is not defined!";
			abortWithErrorMessage(msg);
		}
		constraintFunctionDefinition.executableNameLowFi = executableNameLowFi;
		std::string filenameTrainingDataLowFi;
		filenameTrainingDataLowFi =
				configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex(
						"FILENAME_TRAINING_DATA", 1);
		if (filenameTrainingDataLowFi.empty()) {
			string msg = "FILENAME_TRAINING_DATA is not defined for the low fidelity model";
			abortWithErrorMessage(msg);
		}
		constraintFunctionDefinition.nameLowFidelityTrainingData =filenameTrainingDataLowFi;
		std::string outputFilenameLowFi =
				configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE", 1);
		if (outputFilenameLowFi.empty()) {
			outputFilenameLowFi =
					configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE", 0);
		}
		constraintFunctionDefinition.outputFilenameLowFi = outputFilenameLowFi;
		std::string surrogateModelLowFi;
		surrogateModelLowFi =
				configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL", 1);
		if (surrogateModelLowFi.empty()) {
			surrogateModelLowFi =
					configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL", 0);
		}
		constraintFunctionDefinition.modelLowFi = getSurrogateModelID(
				surrogateModelLowFi);
		std::string exePathLowFi =
				configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("PATH", 1);
		if (exePathLowFi.empty()) {
			exePathLowFi =
					configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("PATH", 0);
		}
		constraintFunctionDefinition.pathLowFi = exePathLowFi;
	}
}

void RoDeODriver::parseConstraintDefinition(std::string inputString){

	std::stringstream iss(inputString);

	std::string definitionBuffer;
	std::string designVectorFilename;
	std::string executableName;
	std::string filenameTrainingData;


	std::string outputFilename;
	std::string exePath;
	std::string surrogateModel;


	configKeysConstraintFunction.parseString(inputString);


#if 0
	printKeywordsConstraint();
#endif


	definitionBuffer = configKeysConstraintFunction.getConfigKeyStringValue("DEFINITION");
	designVectorFilename = configKeysConstraintFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");
	executableName = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);
	outputFilename = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",0);
	exePath = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("PATH",0);
	filenameTrainingData = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",0);
	surrogateModel = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",0);

	ObjectiveFunctionDefinition constraintFunctionDefinition;
	ConstraintDefinition constraintExpression;
	constraintExpression.setDefinition(definitionBuffer);


	constraintFunctionDefinition.designVectorFilename = designVectorFilename;
	constraintFunctionDefinition.executableName = executableName;
	constraintFunctionDefinition.outputFilename = outputFilename;
	constraintFunctionDefinition.path = exePath;
	constraintFunctionDefinition.modelHiFi = getSurrogateModelID(surrogateModel);
	constraintFunctionDefinition.nameHighFidelityTrainingData = filenameTrainingData;


	parseConstraintDefinitionMultiFidelity(constraintFunctionDefinition);
	constraintExpression.ID = numberOfConstraints;
	constraintFunctionDefinition.ifDefined = true;
	numberOfConstraints++;

	constraintDefinitions.push_back(constraintFunctionDefinition);
	constraintExpressions.push_back(constraintExpression);


}

ObjectiveFunctionDefinition RoDeODriver::getObjectiveFunctionDefinition(void) const{
	return definitionObjectiveFunction;
}

ConstraintDefinition RoDeODriver::getConstraintExpression(unsigned int i) const{
	assert(i<constraintExpressions.size());
	return constraintExpressions.at(i);
}

ObjectiveFunctionDefinition RoDeODriver::getConstraintDefinition(unsigned int i) const{

	assert(i<constraintDefinitions.size());
	return constraintDefinitions.at(i);
}




unsigned int RoDeODriver::getDimension(void) const{

	return configKeys.getConfigKeyIntValue("DIMENSION");

}

std::string RoDeODriver::getProblemType(void) const{

	return configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

}
std::string RoDeODriver::getProblemName(void) const{

	return configKeys.getConfigKeyStringValue("PROBLEM_NAME");

}

void RoDeODriver::parseObjectiveFunctionDefinitionMultiFidelity() {
	string multilevel =configKeysObjectiveFunction.getConfigKeyStringValue("MULTILEVEL_MODEL");

	if (checkIfOn(multilevel)) {
		definitionObjectiveFunction.ifMultiLevel = true;
		string executableNameLowFi =
				configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE", 1);
		if (executableNameLowFi.empty()) {
			string msg = "EXECUTABLE for the low fidelity model is not defined!";
			abortWithErrorMessage(msg);
		}
		definitionObjectiveFunction.executableNameLowFi = executableNameLowFi;
		string filenameTrainingDataLowFi;
		filenameTrainingDataLowFi =
				configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA", 1);

		if (filenameTrainingDataLowFi.empty()) {
			string msg = "FILENAME_TRAINING_DATA is not defined for the low fidelity model";
			abortWithErrorMessage(msg);
		}

		definitionObjectiveFunction.nameLowFidelityTrainingData =filenameTrainingDataLowFi;
		string outputFilenameLowFi =
				configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE", 1);
		if (outputFilenameLowFi.empty()) {
			outputFilenameLowFi =
					configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex(
							"OUTPUT_FILE", 0);
		}
		definitionObjectiveFunction.outputFilenameLowFi = outputFilenameLowFi;

		string exePathLowFi =
				configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("PATH", 1);
		if (exePathLowFi.empty()) {
			exePathLowFi =configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("PATH", 0);

		}
		definitionObjectiveFunction.pathLowFi = exePathLowFi;
		string surrogateModelLowFi;
		surrogateModelLowFi =
				configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL", 1);
		if (surrogateModelLowFi.empty()) {
			surrogateModelLowFi =
					configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL", 0);
		}
		definitionObjectiveFunction.modelLowFi = getSurrogateModelID(surrogateModelLowFi);
	}
}

void RoDeODriver::parseObjectiveFunctionDefinition(std::string inputString){

	std::stringstream iss(inputString);

	std::string name;
	std::string designVectorFilename;
	std::string executableName;
	std::string outputFilename;
	std::string exePath;
	std::string surrogateModel;

	std::string filenameTrainingData;

	configKeysObjectiveFunction.parseString(inputString);


#if 0
	printKeywordsObjective();
#endif


	name = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");
	designVectorFilename = configKeysObjectiveFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");

	executableName = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);
	outputFilename = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",0);
	exePath = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("PATH",0);
	filenameTrainingData = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",0);
	surrogateModel = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",0);

	definitionObjectiveFunction.name = name;
	definitionObjectiveFunction.designVectorFilename =  designVectorFilename;

	definitionObjectiveFunction.executableName = executableName;
	definitionObjectiveFunction.path = exePath;
	definitionObjectiveFunction.outputFilename = outputFilename;
	definitionObjectiveFunction.nameHighFidelityTrainingData = filenameTrainingData;
	definitionObjectiveFunction.modelHiFi = getSurrogateModelID(surrogateModel);


	parseObjectiveFunctionDefinitionMultiFidelity();
	definitionObjectiveFunction.ifDefined = true;


}

void RoDeODriver::extractObjectiveFunctionDefinitionFromString(std::string inputString){

	output.printMessage("Extracting objective function definition...");

	std::size_t foundObjectiveFunction = inputString.find("OBJECTIVE_FUNCTION");
	if (foundObjectiveFunction != std::string::npos){

		output.printMessage("Objective function definition is found...");

		std::size_t foundLeftBracket = inputString.find("{", foundObjectiveFunction);


		std::string stringBufferObjectiveFunction;
		std::size_t foundRightBracket = inputString.find("}", foundLeftBracket);
		stringBufferObjectiveFunction.assign(inputString,foundLeftBracket+1,foundRightBracket - foundLeftBracket -1);
		parseObjectiveFunctionDefinition(stringBufferObjectiveFunction);


	}


	std::size_t foundObjectiveFunctionAgain = inputString.find("OBJECTIVE_FUNCTION", foundObjectiveFunction + 18);
	if (foundObjectiveFunctionAgain != std::string::npos){


		std::cout<<"ERROR: Multiple definition of OBJECTIVE_FUNCTION is not allowed\n";
		abort();


	}

}

void RoDeODriver::extractConstraintDefinitionsFromString(std::string inputString){


	std::size_t posConstraintFunction = 0;

	while(1){


		std::size_t foundConstraintFunction = inputString.find("CONSTRAINT_FUNCTION", posConstraintFunction);

		if (foundConstraintFunction != std::string::npos){


			std::size_t foundLeftBracket = inputString.find("{", foundConstraintFunction );


			std::string stringBufferConstraintFunction;

			std::size_t foundRightBracket = inputString.find("}", foundLeftBracket);

			stringBufferConstraintFunction.assign(inputString,foundLeftBracket+1,foundRightBracket - foundLeftBracket -1);

			parseConstraintDefinition(stringBufferConstraintFunction);

			posConstraintFunction =  foundLeftBracket;

		}
		else{

			break;
		}

	}


}

void RoDeODriver::extractConfigDefinitionsFromString(std::string inputString){


	output.printMessage("Extracting configuration parameters...");

	std::stringstream iss(inputString);

	while(iss.good())
	{
		std::string singleLine;
		getline(iss,singleLine,'\n');

		singleLine = removeSpacesFromString(singleLine);
		int indxKeyword = configKeys.searchKeywordInString(singleLine);


		if(indxKeyword != -1){

			ConfigKey temp = configKeys.getConfigKey(indxKeyword);

			std::string keyword = temp.name;
			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			configKeys.assignKeywordValueWithIndex(cleanString,indxKeyword);

		}
	}
}

void RoDeODriver::setConstraintBoxConstraints(ConstraintFunction &constraintFunc) const {

	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");
	Bounds boxConstraints(lb, ub);
	constraintFunc.setParameterBounds(boxConstraints);
}

ConstraintFunction RoDeODriver::setConstraint(unsigned int index) const{


	ConstraintDefinition expression = getConstraintExpression(index);
	ObjectiveFunctionDefinition definition = getConstraintDefinition(index);
	definition.name = expression.constraintName;

	int dim = configKeys.getConfigKeyIntValue("DIMENSION");


	ConstraintFunction constraintFunc;
	constraintFunc.setDimension(dim);

	constraintFunc.setParametersByDefinition(definition);
	constraintFunc.setConstraintDefinition(expression);

	setConstraintBoxConstraints(constraintFunc);
	unsigned int nIterForSurrogateTraining = 10000;


	constraintFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);

	string isWarmStartOn = configKeysConstraintFunction.getConfigKeyStringValue("WARM_START");

	if(checkIfOn(isWarmStartOn)){
		constraintFunc.ifWarmStart	 = true;
	}


	return constraintFunc;

}

void RoDeODriver::setOptimizationFeatures(Optimizer &optimizationStudy) const{

	if(output.ifScreenDisplay){
		optimizationStudy.setDisplayOn();
	}

	string keyword  = "MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS";
	configKeys.abortifConfigKeyIsNotSet(keyword);

	int nFunctionEvals = configKeys.getConfigKeyIntValue(keyword);
	optimizationStudy.setMaximumNumberOfIterations(nFunctionEvals);


	keyword = "MAX_NUMBER_OF_INNER_ITERATIONS";
	if(configKeys.ifConfigKeyIsSet(keyword)){


		int numberOfMaxInnerIterations = configKeys.getConfigKeyIntValue(keyword);
		optimizationStudy.setMaximumNumberOfInnerIterations(numberOfMaxInnerIterations);

	}
}

void RoDeODriver::runOptimization(void){

	Optimizer optimizationStudy = setOptimizationStudy();
	setOptimizationFeatures(optimizationStudy);
	optimizationStudy.performEfficientGlobalOptimization();
}

void RoDeODriver::abortIfProblemNameIsNotDefined(const string &name) {
	if (isEmpty(name)) {
		std::string msg =
				"PROBLEM_NAME is not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}
}

void RoDeODriver::abortIfLowerOrUpperBoundsAreMissing(const vec &lb,
		const vec &ub) {
	if (lb.size() == 0) {
		std::string msg =
				"LOWER_BOUNDS are not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}
	if (ub.size() == 0) {
		std::string msg =
				"UPPER_BOUNDS are not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}
}

void RoDeODriver::addBoundsToOptimizationStudy(Optimizer &optimizationStudy) {
	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");
	abortIfLowerOrUpperBoundsAreMissing(lb, ub);
	Bounds boxConstraints(lb, ub);
	optimizationStudy.setBoxConstraints(boxConstraints);
}

void RoDeODriver::addObjectiveFunctionToOptimizationStudy(
		Optimizer &optimizationStudy) {
	ObjectiveFunction objFunc = setObjectiveFunction();
	optimizationStudy.addObjectFunction(objFunc);
}

void RoDeODriver::addConstraintsToOptimizationStudy(
		Optimizer &optimizationStudy) {
	for (int i = 0; i < numberOfConstraints; i++) {
		ConstraintFunction constraintToAdd = setConstraint(i);
		optimizationStudy.addConstraint(constraintToAdd);
	}
}

void RoDeODriver::addDiscreteParametersIfExistToOptimizationStudy(
		Optimizer &optimizationStudy) {
	vec indicesOfDiscreteVariables = configKeys.getConfigKeyVectorDoubleValue(
			"DISCRETE_VARIABLES");
	vec increments = configKeys.getConfigKeyVectorDoubleValue(
			"DISCRETE_VARIABLES_VALUE_INCREMENTS");
	for (unsigned int i = 0; i < indicesOfDiscreteVariables.size(); i++) {
		optimizationStudy.setParameterToDiscrete(
				int(indicesOfDiscreteVariables(i)), increments(i));
	}
}

Optimizer RoDeODriver::setOptimizationStudy(void) {


	string name = configKeys.getConfigKeyStringValue("PROBLEM_NAME");
	abortIfProblemNameIsNotDefined(name);

	int dim = configKeys.getConfigKeyIntValue("DIMENSION");

	Optimizer optimizationStudy(name, dim);

	addObjectiveFunctionToOptimizationStudy(optimizationStudy);
	addBoundsToOptimizationStudy(optimizationStudy);
	addConstraintsToOptimizationStudy(optimizationStudy);
	addDiscreteParametersIfExistToOptimizationStudy(optimizationStudy);

	return optimizationStudy;


}

SURROGATE_MODEL RoDeODriver::getSurrogateModelID(string modelName) const{

	if(modelName.empty()){
		return ORDINARY_KRIGING;
	}
	if(modelName == "Kriging" ||
	   modelName == "kriging" ||
	   modelName == "KRIGING" ||
	   modelName == "ORDINARY_KRIGING" ||
	   modelName == "ordinary_kriging")

	{
		return ORDINARY_KRIGING;
	}
	if(modelName == "UNIVERSAL_KRIGING" ||  modelName == "universal_kriging") {
		return UNIVERSAL_KRIGING;
	}
	if(modelName == "GRADIENT_ENHANCED" || modelName == "gradient_enhanced") {
		return GRADIENT_ENHANCED;
	}
	if(modelName == "TANGENT_ENHANCED" ||  modelName == "tangent_enhanced" || modelName == "Tangent_enhanced") {
		return TANGENT_ENHANCED;
	}
	return NONE;
}

void RoDeODriver::abortSurrogateModelTestIfNecessaryParametersAreNotDefined() {
	configKeys.abortifConfigKeyIsNotSet("DIMENSION");
	configKeys.abortifConfigKeyIsNotSet("LOWER_BOUNDS");
	configKeys.abortifConfigKeyIsNotSet("UPPER_BOUNDS");
	configKeys.abortifConfigKeyIsNotSet("PROBLEM_NAME");
	configKeys.abortifConfigKeyIsNotSet("FILENAME_TRAINING_DATA");
	configKeys.abortifConfigKeyIsNotSet("FILENAME_TEST_DATA");
}

void RoDeODriver::addBoundsToSurrogateTester(SurrogateModelTester &surrogateTest) {
	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");
	Bounds boxConstraints(lb, ub);
	surrogateTest.setBoxConstraints(boxConstraints);
}

void RoDeODriver::addLowFiModelTypeToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	SURROGATE_MODEL surrogateModelTypeLowFi;
	string surrogateModelLowFi;
	surrogateModelLowFi = configKeys.getConfigKeyStringVectorValueAtIndex(
			"SURROGATE_MODEL", 1);
	surrogateModelTypeLowFi = getSurrogateModelID(surrogateModelLowFi);
	surrogateTest.setSurrogateModelLowFi(surrogateModelTypeLowFi);
}

void RoDeODriver::addLowFiTrainingDataToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	string filenameTrainingDataLowFi;
	filenameTrainingDataLowFi = configKeys.getConfigKeyStringVectorValueAtIndex(
			"FILENAME_TRAINING_DATA", 1);
	surrogateTest.setFileNameTrainingDataLowFidelity(filenameTrainingDataLowFi);
}

void RoDeODriver::addParametersToSurrogateTesterMultiFidelity(SurrogateModelTester &surrogateTest) {

	string multiFidelity = configKeys.getConfigKeyStringValue("MULTILEVEL_MODEL");

	if (checkIfOn(multiFidelity)) {
		addLowFiModelTypeToSurrogateTester(surrogateTest);
		addLowFiTrainingDataToSurrogateTester(surrogateTest);
		surrogateTest.ifMultiLevel = true;
	}
}

void RoDeODriver::addModelTypeToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	SURROGATE_MODEL surrogateModelType;
	string surrogateModelHiFi = configKeys.getConfigKeyStringVectorValueAtIndex(
			"SURROGATE_MODEL", 0);
	surrogateModelType = getSurrogateModelID(surrogateModelHiFi);
	surrogateTest.setSurrogateModel(surrogateModelType);
}

void RoDeODriver::addTrainingDataToSurrogateModelTester(
		SurrogateModelTester &surrogateTest) {
	string filenameTrainingDataHiFi =
			configKeys.getConfigKeyStringVectorValueAtIndex(
					"FILENAME_TRAINING_DATA", 0);
	surrogateTest.setFileNameTrainingData(filenameTrainingDataHiFi);
}

void RoDeODriver::addTestDataToSurrogateModelTester(
		SurrogateModelTester &surrogateTest) {
	string filenameTestData = configKeys.getConfigKeyStringValue(
			"FILENAME_TEST_DATA");
	surrogateTest.setFileNameTestData(filenameTestData);
}

void RoDeODriver::addNumberOfTrainingIterationsToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	if (configKeys.ifConfigKeyIsSet("NUMBER_OF_TRAINING_ITERATIONS")) {
		unsigned int numberOfIterationsForSurrogateModelTraining =
				configKeys.getConfigKeyIntValue(
						"NUMBER_OF_TRAINING_ITERATIONS");
		surrogateTest.setNumberOfTrainingIterations(
				numberOfIterationsForSurrogateModelTraining);
	}
}

void RoDeODriver::addProblemNameToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	string problemName = configKeys.getConfigKeyStringValue("PROBLEM_NAME");
	surrogateTest.setName(problemName);
}

void RoDeODriver::addDimensionToSurrogateTester(
		SurrogateModelTester &surrogateTest) {
	int dimension = configKeys.getConfigKeyIntValue("DIMENSION");
	surrogateTest.setDimension(dimension);
}

void RoDeODriver::runSurrogateModelTest(void){

	abortSurrogateModelTestIfNecessaryParametersAreNotDefined();

	SurrogateModelTester surrogateTest;

	addProblemNameToSurrogateTester(surrogateTest);
	addDimensionToSurrogateTester(surrogateTest);
	addBoundsToSurrogateTester(surrogateTest);
	addTrainingDataToSurrogateModelTester(surrogateTest);
	addModelTypeToSurrogateTester(surrogateTest);
	addTestDataToSurrogateModelTester(surrogateTest);

	addParametersToSurrogateTesterMultiFidelity(surrogateTest);

	surrogateTest.bindSurrogateModels();

	addNumberOfTrainingIterationsToSurrogateTester(surrogateTest);

	string isDisplayOn = configKeys.getConfigKeyStringValue("DISPLAY");

	if(checkIfOn(isDisplayOn)){
		output.ifScreenDisplay = true;
		surrogateTest.setDisplayOn();
	}


	string targetVariableSampleWeights = configKeys.getConfigKeyStringValue("TARGET_VALUE_FOR_VARIABLE_SAMPLE_WEIGHTS");


	string isWarmStartOn = configKeys.getConfigKeyStringValue("WARM_START");
	if(checkIfOn(isWarmStartOn)){
		surrogateTest.ifReadWarmStart = true;
	}

	string msg = "\n################################## STARTING SURROGATE MODEL TEST ##################################\n";
	output.printMessage(msg);
	surrogateTest.performSurrogateModelTest();
	msg = "\n################################## FINISHED SURROGATE MODEL TEST ##################################\n";
	output.printMessage(msg);

}



void RoDeODriver::run(void){

	string isDisplayOn = configKeys.getConfigKeyStringValue("DISPLAY");
	if(checkIfOn(isDisplayOn)){
		output.ifScreenDisplay = true;
	}

	std::string problemType = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

	if(checkifProblemTypeIsSurrogateTest(problemType)){
		runSurrogateModelTest();
	}

	else if(checkifProblemTypeIsOptimization(problemType)){

		string msg = "\n################################## STARTING Optimization ##################################\n";
		output.printMessage(msg);

		runOptimization();

		msg = "\n################################## FINISHED Optimization ##################################\n";
		output.printMessage(msg);
	}

	else{

		abortWithErrorMessage("PROBLEM_TYPE is unknown!");
	}


}


