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
#include <unistd.h>
#include <cassert>
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "auxiliary_functions.hpp"
#include "surrogate_model_tester.hpp"
#include "drivers.hpp"
#include "configkey.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>




RoDeODriver::RoDeODriver(){

	/* Keywords for objective functions */

	configKeysObjectiveFunction.add(ConfigKey("NAME","string") );
	configKeysObjectiveFunction.add(ConfigKey("DESIGN_VECTOR_FILE","string") );

	configKeysObjectiveFunction.add(ConfigKey("OUTPUT_FILE","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("PATH","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("EXECUTABLE","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("SURROGATE_MODEL","stringVector") );

	configKeysObjectiveFunction.add(ConfigKey("FILENAME_TRAINING_DATA","stringVector") );

	configKeysObjectiveFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );
	configKeysObjectiveFunction.add(ConfigKey("MULTILEVEL_MODEL","string") );


	/* Keywords for constraints */

	configKeysConstraintFunction.add(ConfigKey("DEFINITION","string") );
	configKeysConstraintFunction.add(ConfigKey("DESIGN_VECTOR_FILE","string") );

	configKeysConstraintFunction.add(ConfigKey("OUTPUT_FILE","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("PATH","stringVector") );

	configKeysConstraintFunction.add(ConfigKey("EXECUTABLE","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("SURROGATE_MODEL","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );
	configKeysConstraintFunction.add(ConfigKey("MULTILEVEL_MODEL","string") );
	configKeysConstraintFunction.add(ConfigKey("FILENAME_TRAINING_DATA","stringVector") );



	/* Other keywords */
	configKeys.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );

	configKeys.add(ConfigKey("PROBLEM_TYPE","string") );
	configKeys.add(ConfigKey("PROBLEM_NAME","string") );
	configKeys.add(ConfigKey("DIMENSION","int") );
	configKeys.add(ConfigKey("UPPER_BOUNDS","doubleVector") );
	configKeys.add(ConfigKey("LOWER_BOUNDS","doubleVector") );


	configKeys.add(ConfigKey("NUMBER_OF_TEST_SAMPLES","int") );
	configKeys.add(ConfigKey("NUMBER_OF_TRAINING_SAMPLES","int") );


	configKeys.add(ConfigKey("FILENAME_TRAINING_DATA","stringVector") );
	configKeys.add(ConfigKey("FILENAME_TEST_DATA","string") );

	configKeys.add(ConfigKey("SURROGATE_MODEL","string") );

	configKeys.add(ConfigKey("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS","int") );

	configKeys.add(ConfigKey("DESIGN_VECTOR_FILENAME","string") );

	configKeys.add(ConfigKey("VISUALIZATION","string") );
	configKeys.add(ConfigKey("DISPLAY","string") );
	configKeys.add(ConfigKey("ZOOM_IN","string") );
	configKeys.add(ConfigKey("ZOOM_IN_CONTRACTION_FACTOR","double") );
	configKeys.add(ConfigKey("ZOOM_IN_HOW_OFTEN","int") );

	configKeys.add(ConfigKey("MAX_NUMBER_OF_INNER_ITERATIONS","int") );

	configKeys.add(ConfigKey("DISCRETE_VARIABLES","doubleVector") );
	configKeys.add(ConfigKey("DISCRETE_VARIABLES_VALUE_INCREMENTS","doubleVector") );



#if 0
	printKeywords();
#endif

	availableSurrogateModels.push_back("ORDINARY_KRIGING");
	availableSurrogateModels.push_back("UNIVERSAL_KRIGING");
	availableSurrogateModels.push_back("LINEAR_REGRESSION");
	availableSurrogateModels.push_back("AGGREGATION");
	availableSurrogateModels.push_back("MULTI_LEVEL");
	availableSurrogateModels.push_back("TANGENT");

	configFileName = settings.config_file;

}



void RoDeODriver::setDisplayOn(void){

	configKeys.assignKeywordValue("DISPLAY",std::string("ON"));
	output.ifScreenDisplay = true;

}
void RoDeODriver::setDisplayOff(void){

	configKeys.assignKeywordValue("DISPLAY",std::string("OFF"));
	output.ifScreenDisplay = false;
}



void RoDeODriver::readConfigFile(void){

	assert(isNotEmpty(configFileName));

	output.printMessage("Reading the configuration file: ", configFileName);


	std::string stringCompleteFile;

	readFileToaString(configFileName, stringCompleteFile);

	output.printMessage("Configuration file = ");
	output.printMessage(stringCompleteFile);

	extractConfigDefinitionsFromString(stringCompleteFile);

	extractObjectiveFunctionDefinitionFromString(stringCompleteFile);

	extractConstraintDefinitionsFromString(stringCompleteFile);

	checkConsistencyOfConfigParams();
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

	ConfigKey dimension = configKeys.getConfigKey("DIMENSION");


	if(dimension.intValue > 1000){

		std::cout<<"ERROR: Problem dimension is too large, did you set DIMENSION properly?\n";
		abort();

	}

	if(dimension.intValue <= 0){

		std::cout<<"ERROR: Problem dimension must be a positive integer, did you set DIMENSION properly?\n";
		abort();

	}



}



void RoDeODriver::checkIfBoxConstraintsAreSetPropertly(void) const{

	if(ifDisplayIsOn()){

		std::cout<<"Checking box constraint settings...\n";
	}

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


void RoDeODriver::checkIfSurrogateModelTypeIsOK(void) const{

	ConfigKey surrogateModelType = configKeys.getConfigKey("SURROGATE_MODEL");

	surrogateModelType.abortIfNotSet();


	bool ifTypeIsOK = isIntheList(availableSurrogateModels, surrogateModelType.stringValue);

	if(!ifTypeIsOK){

		std::cout<<"ERROR: Surrogate model is not available, did you set SURROGATE_MODEL properly?\n";
		std::cout<<"Available surrogate models:\n";
		printVector(availableSurrogateModels);
		abort();

	}


}

void RoDeODriver::checkIfConstraintsAreProperlyDefined(void) const{


	if(ifDisplayIsOn()){

		std::cout<<"Checking constraint function settings...\n";
	}


	for(auto it = std::begin(constraints); it != std::end(constraints); ++it) {


		if(it->inequalityType != ">" && it->inequalityType != "<"){

			std::cout<<"ERROR: Only inequality constraints (>,<) are allowed in constraint definitions, did you set CONSTRAINT_DEFINITIONS properly?\n";
			abort();

		}

	}

}



void RoDeODriver::checkSettingsForOptimization(void) const{

	checkIfProblemDimensionIsSetProperly();
	checkIfBoxConstraintsAreSetPropertly();
	checkIfConstraintsAreProperlyDefined();


	configKeys.abortifConfigKeyIsNotSet("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

}





void RoDeODriver::checkConsistencyOfConfigParams(void) const{

	if(ifDisplayIsOn()){
		std::cout<<"Checking consistency of the configuration parameters...\n";
	}

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
	if(checkifProblemTypeIsOptimization(s)) return true;

	return false;
}


ObjectiveFunction RoDeODriver::setObjectiveFunction(void) const{

	output.printMessage("Setting the objective function...");

	std::string objFunName = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");
	int dim = configKeys.getConfigKeyIntValue("DIMENSION");

	ObjectiveFunction objFunc;
	objFunc.setDimension(dim);

	objFunc.setParametersByDefinition(objectiveFunction);

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

	return objFunc;


}


void RoDeODriver::printAllConstraintDefinitions(void) const{

	std::cout<<"\nList of all constraint definitions:\n";

	for ( auto i = constraints.begin(); i != constraints.end(); i++ ) {

		i->print();
	}


}

void RoDeODriver::printObjectiveFunctionDefinition(void) const{

	objectiveFunction.print();

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
	filenameTrainingData = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",0);
	surrogateModel = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",0);

	ConstraintDefinition constraintFunctionDefinition;
	constraintFunctionDefinition.setDefinition(definitionBuffer);

	constraintFunctionDefinition.designVectorFilename = designVectorFilename;
	constraintFunctionDefinition.executableName = executableName;
	constraintFunctionDefinition.outputFilename = outputFilename;
	constraintFunctionDefinition.path = exePath;
	constraintFunctionDefinition.modelHiFi = getSurrogateModelID(surrogateModel);
	constraintFunctionDefinition.nameHighFidelityTrainingData = filenameTrainingData;

	std::string multilevel;
	multilevel = configKeysConstraintFunction.getConfigKeyStringValue("MULTILEVEL_MODEL");


	if(checkIfOn(multilevel)){

		constraintFunctionDefinition.ifMultiLevel = true;

		std::string executableNameLowFi = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",1);
		if(executableNameLowFi.empty()){
			abortWithErrorMessage("EXECUTABLE for the low fidelity model is not defined!");
		}
		constraintFunctionDefinition.executableNameLowFi = executableNameLowFi;

		std::string filenameTrainingDataLowFi;
		filenameTrainingDataLowFi = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",1);
		if(filenameTrainingDataLowFi.empty()){
			abortWithErrorMessage("FILENAME_TRAINING_DATA is not defined for the low fidelity model");
		}
		constraintFunctionDefinition.nameLowFidelityTrainingData = filenameTrainingDataLowFi;

		std::string outputFilenameLowFi = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",1);

		if(outputFilenameLowFi.empty()){
			outputFilenameLowFi = outputFilename;
		}
		constraintFunctionDefinition.outputFilenameLowFi = outputFilenameLowFi;

		std::string surrogateModelLowFi;
		surrogateModelLowFi = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",1);
		if(surrogateModelLowFi.empty()){
			surrogateModelLowFi = surrogateModel;
		}
		constraintFunctionDefinition.modelLowFi = getSurrogateModelID(surrogateModelLowFi);

		std::string exePathLowFi = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("PATH",1);
		if(exePathLowFi.empty()){
			exePathLowFi = exePath;
		}
		constraintFunctionDefinition.pathLowFi = exePathLowFi;

	}

	constraintFunctionDefinition.ID = numberOfConstraints;
	numberOfConstraints++;

	constraints.push_back(constraintFunctionDefinition);


}

ObjectiveFunctionDefinition RoDeODriver::getObjectiveFunctionDefinition(void) const{
	return objectiveFunction;
}

ConstraintDefinition RoDeODriver::getConstraintDefinition(unsigned int i) const{
	assert(i<constraints.size());
	return constraints.at(i);
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

void RoDeODriver::parseObjectiveFunctionDefinition(std::string inputString){

	std::stringstream iss(inputString);

	std::string name;
	std::string designVectorFilename;
	std::string executableName;
	std::string outputFilename;
	std::string exePath;
	std::string surrogateModel;
	std::string multilevel;
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

	objectiveFunction.name = name;
	objectiveFunction.designVectorFilename =  designVectorFilename;

	objectiveFunction.executableName = executableName;
	objectiveFunction.path = exePath;
	objectiveFunction.outputFilename = outputFilename;
	objectiveFunction.nameHighFidelityTrainingData = filenameTrainingData;
	objectiveFunction.modelHiFi = getSurrogateModelID(surrogateModel);


	multilevel = configKeysObjectiveFunction.getConfigKeyStringValue("MULTILEVEL_MODEL");


	if(checkIfOn(multilevel)){

		objectiveFunction.ifMultiLevel = true;

		std::string executableNameLowFi = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",1);
		if(executableNameLowFi.empty()){
			abortWithErrorMessage("EXECUTABLE for the low fidelity model is not defined!");
		}
		objectiveFunction.executableNameLowFi = executableNameLowFi;

		std::string filenameTrainingDataLowFi;

		filenameTrainingDataLowFi = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",1);

		if(filenameTrainingDataLowFi.empty()){
			abortWithErrorMessage("FILENAME_TRAINING_DATA is not defined for the low fidelity model");
		}
		objectiveFunction.nameLowFidelityTrainingData = filenameTrainingDataLowFi;


		std::string outputFilenameLowFi = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",1);
		if(outputFilenameLowFi.empty()){
			outputFilenameLowFi = outputFilename;
		}
		objectiveFunction.outputFilenameLowFi = outputFilenameLowFi;

		std::string exePathLowFi = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("PATH",1);
		if(exePathLowFi.empty()){
			exePathLowFi = exePath;
		}
		objectiveFunction.pathLowFi = exePathLowFi;

		std::string surrogateModelLowFi;
		surrogateModelLowFi = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("SURROGATE_MODEL",1);
		if(surrogateModelLowFi.empty()){
			surrogateModelLowFi = surrogateModel;
		}
		objectiveFunction.modelLowFi = getSurrogateModelID(surrogateModelLowFi);


	}
	objectiveFunction.ifDefined = true;



}

void RoDeODriver::extractObjectiveFunctionDefinitionFromString(std::string inputString){

	if(ifDisplayIsOn()){

		std::cout<<"Extracting objective function definition\n";

	}

	std::size_t foundObjectiveFunction = inputString.find("OBJECTIVE_FUNCTION");
	if (foundObjectiveFunction != std::string::npos){

		if(ifDisplayIsOn()){

			std::cout<<"Objective function definition is found\n";

		}


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



			displayMessage("Constraint function definition is found:");
			displayMessage("\nDefinition begin");
			displayMessage(stringBufferConstraintFunction);
			displayMessage("\nDefinition end");


			parseConstraintDefinition(stringBufferConstraintFunction);

			posConstraintFunction =  foundLeftBracket;

		}
		else{

			break;
		}

	}


}

void RoDeODriver::extractConfigDefinitionsFromString(std::string inputString){


	displayMessage("Extracting configuration parameters...");


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


	displayMessage("Extracting configuration parameters is done...");


}



ConstraintFunction RoDeODriver::setConstraint(ConstraintDefinition constraintDefinition) const{



	int dim = configKeys.getConfigKeyIntValue("DIMENSION");


	ConstraintFunction constraintFunc;
	constraintFunc.setDimension(dim);

	constraintFunc.setParametersByDefinition(constraintDefinition);

	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	Bounds boxConstraints(lb,ub);
	constraintFunc.setParameterBounds(boxConstraints);



	unsigned int nIterForSurrogateTraining = 10000;


	constraintFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);

	return constraintFunc;

}

void RoDeODriver::setOptimizationFeatures(Optimizer &optimizationStudy) const{

	if(ifDisplayIsOn()){
		optimizationStudy.setDisplayOn();
	}

	string keyword  = "MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS";
	configKeys.abortifConfigKeyIsNotSet(keyword);

	int nFunctionEvals = configKeys.getConfigKeyIntValue(keyword);
	optimizationStudy.setMaximumNumberOfIterations(nFunctionEvals);


	keyword = "MAX_NUMBER_OF_INNER_ITERATIONS";
	if(configKeys.ifConfigKeyIsSet(keyword)){

		int numberOfMaxInnerIterations = configKeys.getConfigKeyIntValue("keyword");
		optimizationStudy.setMaximumNumberOfInnerIterations(numberOfMaxInnerIterations);

	}



}

void RoDeODriver::runOptimization(void){

	output.printMessage("Running Optimization...");

	Optimizer optimizationStudy = setOptimizationStudy();
	setOptimizationFeatures(optimizationStudy);

	if(output.ifScreenDisplay){
		optimizationStudy.print();
	}

	optimizationStudy.EfficientGlobalOptimization();

}

bool RoDeODriver::ifIsAGradientBasedMethod(std::string modelType) const{

	if(modelType == "GRADIENT_ENHANCED_KRIGING" || modelType == "AGGREGATION"){

		return true;
	}
	else return false;


}



void RoDeODriver::determineProblemDimensionAndBoxConstraintsFromTrainingData(void){

	/* These two are absolutely required */
	configKeys.abortifConfigKeyIsNotSet("FILENAME_TRAINING_DATA");
	configKeys.abortifConfigKeyIsNotSet("SURROGATE_MODEL");


	std::string fileName = configKeys.getConfigKeyStringValue("FILENAME_TRAINING_DATA");
	std::string surrogateModelType = configKeys.getConfigKeyStringValue("SURROGATE_MODEL");



	mat bufferData;
	bufferData.load(fileName, csv_ascii);

	int dim;

	unsigned int nCols = bufferData.n_cols;

	if(  ifIsAGradientBasedMethod(surrogateModelType)){

		dim = (nCols -1)/2;

	}
	else{

		dim = (nCols -1);

	}

	assert(dim>0);


	configKeys.assignKeywordValue("DIMENSION",dim);

	vec lb,ub;


	lb = zeros<vec>(dim);
	ub = zeros<vec>(dim);

	for(unsigned int i=0; i<dim; i++){

		vec columnOfData = bufferData.col(i);

		lb(i) = min(columnOfData);
		ub(i) = max(columnOfData);

	}


	configKeys.assignKeywordValue("LOWER_BOUNDS",lb);
	configKeys.assignKeywordValue("UPPER_BOUNDS",ub);


	if(ifDisplayIsOn()){


		printVector(lb,"boxConstraintsLowerBounds");
		printVector(ub,"boxConstraintsUpperBounds");
		std::cout<<"dimension = "<<dim<<"\n";

	}

}

Optimizer RoDeODriver::setOptimizationStudy(void) {


	string name = configKeys.getConfigKeyStringValue("PROBLEM_NAME");

	if(isEmpty(name)){

		std::string msg = "PROBLEM_NAME is not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}

	int dim = configKeys.getConfigKeyIntValue("DIMENSION");

	Optimizer optimizationStudy(name, dim);
	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	if(lb.size() == 0){

		std::string msg = "LOWER_BOUNDS are not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}
	if(ub.size() == 0){

		std::string msg = "UPPER_BOUNDS are not set! Check the configuration file.";
		abortWithErrorMessage(msg);
	}


	ObjectiveFunction objFunc = setObjectiveFunction();
	optimizationStudy.addObjectFunction(objFunc);

	Bounds boxConstraints(lb,ub);
	optimizationStudy.setBoxConstraints(boxConstraints);


	for ( auto i = constraints.begin(); i != constraints.end(); i++ ) {

		ConstraintFunction constraintToAdd = setConstraint(*i);
		optimizationStudy.addConstraint(constraintToAdd);

	}


	vec indicesOfDiscreteVariables = configKeys.getConfigKeyVectorDoubleValue("DISCRETE_VARIABLES");
	vec increments = configKeys.getConfigKeyVectorDoubleValue("DISCRETE_VARIABLES_VALUE_INCREMENTS");

	for(unsigned int i = 0; i<indicesOfDiscreteVariables.size(); i++){

		optimizationStudy.setParameterToDiscrete(int(indicesOfDiscreteVariables(i)), increments(i));
	}




	return optimizationStudy;


}



SURROGATE_MODEL RoDeODriver::getSurrogateModelID(string modelName) const{

	if(modelName.empty()){

		return ORDINARY_KRIGING;
	}

	if(modelName == "Kriging" || modelName == "KRIGING" || modelName == "ORDINARY_KRIGING" ||  modelName == "ordinary_kriging") {

		return ORDINARY_KRIGING;

	}

	if(modelName == "UNIVERSAL_KRIGING" ||  modelName == "universal_kriging") {

		return UNIVERSAL_KRIGING;

	}

	if(modelName == "GEK" ||  modelName == "GRADIENT_ENHANCED_KRIGING" || modelName == "gek") {

		return GRADIENT_ENHANCED_KRIGING;

	}

	if(modelName == "AGGREGATION_MODEL" ||  modelName == "AGGREGATION" || modelName == "aggregation_model") {

		return AGGREGATION;

	}

	if(modelName == "MULTI_LEVEL" ||  modelName == "multi-level" || modelName == "MULTI-LEVEL") {
		return MULTI_LEVEL;
	}

	if(modelName == "TANGENT" ||  modelName == "tangent" || modelName == "Tangent") {
		return TANGENT;
	}



	return NONE;
}

void RoDeODriver::runSurrogateModelTest(void){

	SurrogateModelTester surrogateTest;

	std::string problemName = configKeys.getConfigKeyStringValue("PROBLEM_NAME");
	surrogateTest.setName(problemName);

	std::string surrogateModelType = configKeys.getConfigKeyStringValue("SURROGATE_MODEL");
	SURROGATE_MODEL modelID = getSurrogateModelID(surrogateModelType);

	int dimension = configKeys.getConfigKeyIntValue("DIMENSION");
	surrogateTest.setDimension(dimension);

	std::string filenameTrainingData = configKeys.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",0);
	surrogateTest.setFileNameTrainingData(filenameTrainingData);


	if(modelID == MULTI_LEVEL){

		std::string filenameTrainingDataLowFidelity = configKeys.getConfigKeyStringVectorValueAtIndex("FILENAME_TRAINING_DATA",1);
		surrogateTest.setFileNameTrainingDataLowFidelity(filenameTrainingDataLowFidelity);
	}


	std::string filenameTestData = configKeys.getConfigKeyStringValue("FILENAME_TEST_DATA");
	surrogateTest.setFileNameTestData(filenameTestData);

	if(configKeys.ifConfigKeyIsSet("LOWER_BOUNDS") && configKeys.ifConfigKeyIsSet("UPPER_BOUNDS")){

		vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
		vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

		Bounds boxConstraints(lb,ub);
		surrogateTest.setBoxConstraints(boxConstraints);

	}

	if(configKeys.ifConfigKeyIsSet("NUMBER_OF_TRAINING_ITERATIONS")){

		unsigned int numberOfIterationsForSurrogateModelTraining = configKeys.getConfigKeyIntValue("NUMBER_OF_TRAINING_ITERATIONS");

		surrogateTest.setNumberOfTrainingIterations(numberOfIterationsForSurrogateModelTraining);

	}


	surrogateTest.setSurrogateModel(modelID);

	if(configKeys.ifConfigKeyIsSet("DISPLAY")) {

		std::string display = configKeys.getConfigKeyStringValue("DISPLAY");

		if(checkIfOn(display)){
			surrogateTest.setDisplayOn();
		}
	}

	surrogateTest.performSurrogateModelTest();


}



void RoDeODriver::run(void){


	std::string problemType = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

	if(checkifProblemTypeIsSurrogateTest(problemType)){

		string msg = "\n################################## STARTING SURROGATE MODEL TEST ##################################\n";
		output.printMessage(msg);

		runSurrogateModelTest();

		msg = "\n################################## FINISHED SURROGATE MODEL TEST ##################################\n";
		output.printMessage(msg);

	}

	if(checkifProblemTypeIsOptimization(problemType)){

		string msg = "\n################################## STARTING Optimization ##################################\n";
		output.printMessage(msg);

		runOptimization();

		msg = "\n################################## FINISHED Optimization ##################################\n";
		output.printMessage(msg);
	}


	abortWithErrorMessage("PROBLEM_TYPE is unknown!");

}



void RoDeODriver::displayMessage(std::string inputString) const{

	if(ifDisplayIsOn()){

		std::cout<<inputString<<"\n";


	}


}

bool RoDeODriver::ifDisplayIsOn(void) const{


	if(configKeys.ifFeatureIsOn("DISPLAY")){

		return true;


	}
	else{

		return false;

	}


}


