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
	configKeysObjectiveFunction.add(ConfigKey("GRADIENT","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("EXECUTABLE","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("MARKER","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("MARKER_FOR_GRADIENT","stringVector") );
	configKeysObjectiveFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );
	configKeysObjectiveFunction.add(ConfigKey("MULTILEVEL_SURROGATE","string") );

	/* Keywords for constraints */

	configKeysConstraintFunction.add(ConfigKey("DEFINITION","string") );
	configKeysConstraintFunction.add(ConfigKey("DESIGN_VECTOR_FILE","string") );

	configKeysConstraintFunction.add(ConfigKey("OUTPUT_FILE","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("PATH","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("GRADIENT","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("EXECUTABLE","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("MARKER","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("MARKER_FOR_GRADIENT","stringVector") );
	configKeysConstraintFunction.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );
	configKeysConstraintFunction.add(ConfigKey("MULTILEVEL_SURROGATE","string") );



	/* Other keywords */
	configKeys.add(ConfigKey("PROBLEM_TYPE","string") );
	configKeys.add(ConfigKey("PROBLEM_NAME","string") );
	configKeys.add(ConfigKey("DIMENSION","int") );
	configKeys.add(ConfigKey("UPPER_BOUNDS","doubleVector") );
	configKeys.add(ConfigKey("LOWER_BOUNDS","doubleVector") );


	configKeys.add(ConfigKey("NUMBER_OF_TEST_SAMPLES","int") );
	configKeys.add(ConfigKey("NUMBER_OF_TRAINING_SAMPLES","int") );


	configKeys.add(ConfigKey("FILENAME_TRAINING_DATA","string") );
	configKeys.add(ConfigKey("FILENAME_TEST_DATA","string") );

	configKeys.add(ConfigKey("SURROGATE_MODEL","string") );

	configKeys.add(ConfigKey("NUMBER_OF_DOE_SAMPLES","int") );
	configKeys.add(ConfigKey("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS","int") );

	configKeys.add(ConfigKey("WARM_START","string") );
	configKeys.add(ConfigKey("DESIGN_VECTOR_FILENAME","string") );

	configKeys.add(ConfigKey("VISUALIZATION","string") );
	configKeys.add(ConfigKey("DISPLAY","string") );

	configKeys.add(ConfigKey("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION","int") );


#if 0
	printKeywords();
#endif

	availableSurrogateModels.push_back("ORDINARY_KRIGING");
	availableSurrogateModels.push_back("UNIVERSAL_KRIGING");
	availableSurrogateModels.push_back("GRADIENT_ENHANCED_KRIGING");
	availableSurrogateModels.push_back("LINEAR_REGRESSION");
	availableSurrogateModels.push_back("AGGREGATION");

	configFileName = settings.config_file;

}



void RoDeODriver::setDisplayOn(void){



	configKeys.assignKeywordValue("DISPLAY",std::string("ON"));


}
void RoDeODriver::setDisplayOff(void){


	configKeys.assignKeywordValue("DISPLAY",std::string("OFF"));
}







std::string RoDeODriver::removeKeywordFromString(std::string inputStr,  std::string keyword) const{

	assert(!keyword.empty());
	assert(!inputStr.empty());

	std::size_t found = inputStr.find(keyword);

	if(found != std::string::npos){

		inputStr.erase(std::remove_if(inputStr.begin(), inputStr.end(), isspace), inputStr.end());
		std::string sub_str = inputStr.substr(found+keyword.length() + 1);


		return sub_str;


	}
	else{

		return inputStr;
	}



}





void RoDeODriver::readConfigFile(void){


	if(ifDisplayIsOn()){

		std::cout<<"Reading the configuration file : "<< configFileName <<"\n";

	}

	std::string stringCompleteFile;

	readFileToaString(configFileName, stringCompleteFile);

	if(ifDisplayIsOn()){

		std::cout<<"Configuration file = \n";
		std::cout<<stringCompleteFile;

	}

	extractConfigDefinitionsFromString(stringCompleteFile);

	extractObjectiveFunctionDefinitionFromString(stringCompleteFile);

	extractConstraintDefinitionsFromString(stringCompleteFile);





#if 0
	printKeywords();
#endif

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


//void RoDeODriver::checkIfObjectiveFunctionIsSetProperly(void) const{
//
//	ConfigKey configKeyName = getConfigKeyObjective("NAME");
//	configKeyName.abortIfNotSet();
//
//
//}




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

		std::cout<<"ERROR: Dimension of bounds does not match with the problem dimension, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?\n";
		abort();

	}

	for(int i=0; i<dim; i++){

		if( ub.vectorDoubleValue(i) <= lb.vectorDoubleValue(i)){

			std::cout<<"ERROR: Lower bounds cannot be greater or equal than upper bounds, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?\n";
			abort();


		}


	}


}






void RoDeODriver::checkSettingsForSurrogateModelTest(void) const{


	checkIfSurrogateModelTypeIsOK();

	bool ifTrainingDataIsSet = configKeys.ifConfigKeyIsSet("FILENAME_TRAINING_DATA");
	bool ifTestDataIsSet = configKeys.ifConfigKeyIsSet("FILENAME_TEST_DATA");


	if(ifTrainingDataIsSet == false || ifTestDataIsSet== false ){

		checkIfProblemDimensionIsSetProperly();

		checkIfBoxConstraintsAreSetPropertly();

	}


	if(ifTrainingDataIsSet == false) {

		configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");

	}

	if(ifTestDataIsSet == false) {

		configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");

	}




}


//
//void RoDeODriver::checkIfNumberOfTrainingSamplesIsDefined(void) const{
//
//	configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");
//
//}
//
//void RoDeODriver::checkIfNumberOfTestSamplesIsDefined(void) const{
//
//	configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");
//
//}




void RoDeODriver::checkIfSurrogateModelTypeIsOK(void) const{

	ConfigKey surrogateModelType = configKeys.getConfigKey("SURROGATE_MODEL");

	surrogateModelType.abortIfNotSet();


	bool ifTypeIsOK = ifIsInTheList(availableSurrogateModels, surrogateModelType.stringValue);

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


void RoDeODriver::checkSettingsForDoE(void) const{

	checkIfProblemDimensionIsSetProperly();
	checkIfBoxConstraintsAreSetPropertly();

	checkIfConstraintsAreProperlyDefined();

	configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");


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

	checkIfObjectiveFunctionNameIsDefined();


	std::string type = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");


	if(type == "SURROGATE_TEST"){

		checkSettingsForSurrogateModelTest();



	}

	if(type == "DoE"){

		checkSettingsForDoE();

	}




	if(type == "Minimization" || type == "Maximization"){

		checkSettingsForOptimization();


	}


}

void RoDeODriver::checkIfProblemTypeIsSetProperly(void) const{


	ConfigKey problemType = configKeys.getConfigKey("PROBLEM_TYPE");
	problemType.abortIfNotSet();

	bool ifProblemTypeIsValid = checkifProblemTypeIsValid(problemType.stringValue);

	if(!ifProblemTypeIsValid){

		std::cout<<"ERROR: Problem type is not valid, did you set PROBLEM_TYPE properly?\n";
		std::cout<<"Valid problem types: MINIMIZATION, MAXIMIZATION, DoE, SURROGATE_TEST\n";
		abort();

	}

}




bool RoDeODriver::checkifProblemTypeIsValid(std::string s) const{

	if (s == "DoE" || s == "MINIMIZATION" || s == "MAXIMIZATION" || s == "SURROGATE_TEST"){

		return true;
	}
	else return false;


}


ObjectiveFunction RoDeODriver::setObjectiveFunction(void) const{

	if(ifDisplayIsOn()){

		std::cout<<"Setting objective function...\n";

	}



	std::string objFunName = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");
	int dim = configKeys.getConfigKeyIntValue("DIMENSION");
	ObjectiveFunction objFunc(objFunName, dim);


	objFunc.setParametersByDefinition(objectiveFunction);


	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	objFunc.setParameterBounds(lb,ub);


	std::string ifGradientAvailable = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("GRADIENT",0);


	if(checkIfOn(ifGradientAvailable)){

		objFunc.setGradientOn();

	}


	unsigned int nIterForSurrogateTraining = 10000;

	if(configKeysObjectiveFunction.ifConfigKeyIsSet("NUMBER_OF_TRAINING_ITERATIONS")){

		nIterForSurrogateTraining = configKeysObjectiveFunction.getConfigKeyIntValue("NUMBER_OF_TRAINING_ITERATIONS");

	}


	objFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);

	if(ifDisplayIsOn()){

		objFunc.print();

	}


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


	std::string outputFilename;
	std::string exePath;
	std::string marker;
	std::string markerGradient;
	std::string ifGradient;


	while(iss.good())
	{
		std::string singleLine;
		getline(iss,singleLine,'\n');

		singleLine = removeSpacesFromString(singleLine);
		int indxKeyword = configKeysConstraintFunction.searchKeywordInString(singleLine);


		if(indxKeyword != -1){

			ConfigKey temp = configKeysConstraintFunction.getConfigKey(indxKeyword);

			std::string keyword = temp.name;
			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			configKeysConstraintFunction.assignKeywordValueWithIndex(cleanString,indxKeyword);


		}


	}

#if 0
	printKeywordsConstraint();
#endif


	definitionBuffer = configKeysConstraintFunction.getConfigKeyStringValue("DEFINITION");
	designVectorFilename = configKeysConstraintFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");




	executableName = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);
	outputFilename = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",0);
	exePath = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("PATH",0);
	marker = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("MARKER",0);
	markerGradient = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("MARKER_FOR_GRADIENT",0);
	ifGradient = configKeysConstraintFunction.getConfigKeyStringVectorValueAtIndex("GRADIENT",0);




	ConstraintDefinition result(definitionBuffer);
	result.designVectorFilename = designVectorFilename;
	result.executableName = executableName;
	result.outputFilename = outputFilename;
	result.path = exePath;
	result.marker = marker;
	result.markerForGradient = markerGradient;

	result.ID = numberOfConstraints;
	numberOfConstraints++;


	if(checkIfOn(ifGradient)){

		result.ifGradient = true;

	}

	if(ifDisplayIsOn()){

		std::cout <<"Adding a constraint definition with ID = "<< result.ID<<"\n";
	}


	constraints.push_back(result);


}


ObjectiveFunctionDefinition RoDeODriver::getObjectiveFunctionDefinition(void) const{

	return objectiveFunction;


}


ConstraintDefinition RoDeODriver::getConstraintDefinition(unsigned int i) const{

	assert(i<constraints.size());
	return this->constraints.at(i);


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
	std::string gradient;
	std::string marker;
	std::string markerGradient;

	std::string multilevel;

	while(iss.good())
	{
		std::string singleLine;
		getline(iss,singleLine,'\n');

		singleLine = removeSpacesFromString(singleLine);


		int indxKeyword = configKeysObjectiveFunction.searchKeywordInString(singleLine);


		if(indxKeyword != -1){

			ConfigKey temp = configKeysObjectiveFunction.getConfigKey(indxKeyword);

			std::string keyword = temp.name;


			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			configKeysObjectiveFunction.assignKeywordValueWithIndex(cleanString,indxKeyword);


		}


	}

#if 0
	printKeywordsObjective();
#endif



	name = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");
	designVectorFilename = configKeysObjectiveFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");


	executableName = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);
	outputFilename = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",0);
	exePath = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("PATH",0);
	marker =  configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("MARKER",0);
	markerGradient = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("MARKER_FOR_GRADIENT",0);
	gradient = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("GRADIENT",0);

	multilevel = configKeysObjectiveFunction.getConfigKeyStringValue("MULTILEVEL_SURROGATE");


	objectiveFunction.name = name;
	objectiveFunction.designVectorFilename =  designVectorFilename;

	objectiveFunction.executableName = executableName;
	objectiveFunction.path = exePath;
	objectiveFunction.outputFilename = outputFilename;
	objectiveFunction.marker = marker;
	objectiveFunction.markerForGradient = markerGradient;



	if(checkIfOn(gradient)){

		objectiveFunction.ifGradient = true;

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
	else{

		std::cout<<"ERROR: OBJECTIVE_FUNCTION is absent!\n";
		abort();

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


}



ConstraintFunction RoDeODriver::setConstraint(ConstraintDefinition constraintDefinition) const{

	assert(constraintDefinition.ID >= 0);

	if(ifDisplayIsOn()){

		std::cout<<"Setting the constraint with ID = "<< constraintDefinition.ID << "\n";

	}


	int dim = configKeys.getConfigKeyIntValue("DIMENSION");


	ConstraintFunction constraintFunc(constraintDefinition.name, dim);
	constraintFunc.setParametersByDefinition(constraintDefinition);

	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");
	constraintFunc.setParameterBounds(lb,ub);


	if(constraintDefinition.ifGradient){

		constraintFunc.setGradientOn();

	}


	unsigned int nIterForSurrogateTraining = 10000;






	constraintFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);



	if(ifDisplayIsOn()){


		constraintFunc.print();

	}

	if(this->checkIfRunIsNecessary(constraintDefinition.ID)){

		constraintFunc.setrunOn();
	}
	else{

		constraintFunc.setrunOff();
	}



	return constraintFunc;

}

void RoDeODriver::setOptimizationFeatures(Optimizer &optimizationStudy) const{

	if(ifDisplayIsOn()){

		optimizationStudy.setDisplayOn();

	}

	configKeys.abortifConfigKeyIsNotSet("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

	int nFunctionEvals = configKeys.getConfigKeyIntValue("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

	optimizationStudy.setMaximumNumberOfIterations(nFunctionEvals);


	int nIterationsForEIMaximization = configKeys.getConfigKeyIntValue("DIMENSION")* 100000;

	if(configKeys.ifConfigKeyIsSet("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION")){


		nIterationsForEIMaximization = configKeys.getConfigKeyIntValue("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION");

		optimizationStudy.setMaximumNumberOfIterationsForEIMaximization(nIterationsForEIMaximization);

	}



}

void RoDeODriver::runOptimization(void){

	displayMessage("Running Optimization...\n");

	Optimizer optimizationStudy = setOptimizationStudy();

	setOptimizationFeatures(optimizationStudy);


	std::string WarmStart = "OFF";

	if(configKeys.ifConfigKeyIsSet("WARM_START")){


		WarmStart = configKeys.getConfigKeyStringValue("WARM_START");

		std::string msg = "Warm start = " + WarmStart;
		displayMessage(msg);


	}

	if(ifDisplayIsOn()){

		optimizationStudy.print();

	}



	if(checkIfOff(WarmStart)){

		configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");
		int maximumNumberDoESamples = configKeys.getConfigKeyIntValue("NUMBER_OF_DOE_SAMPLES");
		optimizationStudy.cleanDoEFiles();
		optimizationStudy.performDoE(maximumNumberDoESamples,LHS);

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

	std::string name = configKeys.getConfigKeyStringValue("PROBLEM_NAME");
	std::string type = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");
	int dim = configKeys.getConfigKeyIntValue("DIMENSION");

	Optimizer optimizationStudy(name, dim, type);
	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	optimizationStudy.setBoxConstraints(lb,ub);

	std::string dvFilename = configKeysObjectiveFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");

	optimizationStudy.setFileNameDesignVector(dvFilename);


	ObjectiveFunction objFunc = setObjectiveFunction();
	optimizationStudy.addObjectFunction(objFunc);


	for ( auto i = constraints.begin(); i != constraints.end(); i++ ) {

		ConstraintFunction constraintToAdd = setConstraint(*i);
		optimizationStudy.addConstraint(constraintToAdd);

	}



	return optimizationStudy;


}



void RoDeODriver::runDoE(void) {


	Optimizer optimizationStudy = setOptimizationStudy();


	if(!optimizationStudy.checkSettings()){

		std::cout<<"ERROR: Settings check is failed!\n";
		abort();
	}

	bool ifWarmStart = false;


	if(ifDisplayIsOn()){

		ifWarmStart = true;

	}

	if(ifWarmStart == false){

		optimizationStudy.cleanDoEFiles();

	}


	configKeys.abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");
	int maximumNumberDoESamples = configKeys.getConfigKeyIntValue("NUMBER_OF_DOE_SAMPLES");

	optimizationStudy.performDoE(maximumNumberDoESamples,LHS);



}


bool RoDeODriver::checkIfRunIsNecessary(int idConstraint) const{
#if 0
	std::cout<<"checkIfRunIsNecessary with idConstraint = "<<idConstraint<<"\n";
#endif
	assert(objectiveFunction.ifDefined);
	assert(!objectiveFunction.executableName.empty());
	assert(idConstraint < constraints.size());
	assert(idConstraint >= 0);

	ConstraintDefinition constraintToCheck = this->constraints.at(idConstraint);
	std::string exeNameOftheConstraint = constraintToCheck.executableName;

#if 0
	constraintToCheck.print();
#endif

	assert(!exeNameOftheConstraint.empty());


	if(objectiveFunction.executableName == exeNameOftheConstraint) {

		return false;
	}


	if(ifDisplayIsOn()){

		std::cout<<"Checking if a run is necessary for the constraint with ID = "<<idConstraint<<"\n";
	}

	for ( auto i = constraints.begin(); i != constraints.end(); i++ ) {

		if(i->executableName == exeNameOftheConstraint && i->ID < idConstraint) return false;

	}


	return true;

}




void RoDeODriver::runSurrogateModelTest(void){

	if(configKeys.ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		determineProblemDimensionAndBoxConstraintsFromTrainingData();

	}

	std::string surrogateModelType = configKeys.getConfigKeyStringValue("SURROGATE_MODEL");



	std::string objectiveFunctionName = configKeysObjectiveFunction.getConfigKeyStringValue("NAME");

	int dimension = configKeys.getConfigKeyIntValue("DIMENSION");

	TestFunction TestFunction(objectiveFunctionName, dimension);

	if(  ifIsAGradientBasedMethod(surrogateModelType)){

		TestFunction.setGradientsOn();

	}

	if(configKeysObjectiveFunction.ifConfigKeyIsSet("NUMBER_OF_TRAINING_ITERATIONS")){

		int nIter = configKeysObjectiveFunction.getConfigKeyIntValue("NUMBER_OF_TRAINING_ITERATIONS");
		assert(nIter>0);
		assert(nIter<1000000);
		TestFunction.setNumberOfTrainingIterations(nIter);
	}



	if(configKeys.ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		std::string inputFilename = configKeys.getConfigKeyStringValue("FILENAME_TRAINING_DATA");
		assert(!inputFilename.empty());
		TestFunction.setNameFilenameTrainingData(inputFilename);

	}
	else{

		std::string exeName = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);

		TestFunction.setNameOfExecutable(exeName);

		std::string designVectorFilename = configKeysObjectiveFunction.getConfigKeyStringValue("DESIGN_VECTOR_FILE");

		TestFunction.setNameOfInputForExecutable(designVectorFilename);

		std::string outputFilename = configKeysObjectiveFunction.getConfigKeyStringVectorValueAtIndex("OUTPUT_FILE",0);
		TestFunction.setNameOfOutputForExecutable(outputFilename);

		int numberOfTrainingSamples =  configKeys.getConfigKeyIntValue("NUMBER_OF_TRAINING_SAMPLES");
		assert(numberOfTrainingSamples>0);
		TestFunction.setNumberOfTrainingSamples(numberOfTrainingSamples);



	}

	if(configKeys.ifConfigKeyIsSet("FILENAME_TEST_DATA")){

		std::string inputFilename = configKeys.getConfigKeyStringValue("FILENAME_TEST_DATA");
		assert(!inputFilename.empty());
		TestFunction.setNameFilenameTestData(inputFilename);


	}
	else{

		int numberOfTestSamples =  configKeys.getConfigKeyIntValue("NUMBER_OF_TEST_SAMPLES");
		assert(numberOfTestSamples>0);
		TestFunction.setNumberOfTestSamples(numberOfTestSamples);

	}



	vec lb = configKeys.getConfigKeyVectorDoubleValue("LOWER_BOUNDS");
	vec ub = configKeys.getConfigKeyVectorDoubleValue("UPPER_BOUNDS");

	TestFunction.setBoxConstraints(lb, ub);


	if(configKeys.ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		TestFunction.readFileTrainingData();

	}
	else{

		TestFunction.generateSamplesInputTrainingData();
		TestFunction.generateTrainingSamples();

	}

	if(configKeys.ifConfigKeyIsSet("FILENAME_TEST_DATA")){

		TestFunction.readFileTestData();

	}
	else{

		TestFunction.generateSamplesInputTestData();
		TestFunction.generateTestSamples();
	}



	if(configKeys.ifConfigKeyIsSet("VISUALIZATION")) {

		std::string flag = configKeys.getConfigKeyStringValue("VISUALIZATION");

		if(flag == "ON" || flag == "YES"){

			TestFunction.setVisualizationOn();
		}

	}


	if(configKeys.ifConfigKeyIsSet("DISPLAY")) {

		std::string flag = configKeys.getConfigKeyStringValue("DISPLAY");

		if(flag == "ON" || flag == "YES"){

			TestFunction.setDisplayOn();
			TestFunction.print();
		}



	}


#if 0
	TestFunction.print();
#endif


	TestFunction.testSurrogateModel(surrogateModelType);


}



int RoDeODriver::runDriver(void){

	std::string problemType = configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

	if(problemType == "SURROGATE_TEST"){


		std::cout<<"\n################################## STARTING SURROGATE MODEL TEST ##################################\n";
		runSurrogateModelTest();

		std::cout<<"\n################################## FINISHED SURROGATE MODEL TEST ##################################\n";

		return 0;
	}


	if(problemType == "DoE"){


		std::cout<<"\n################################## STARTING DoE ##################################\n";
		runDoE();

		std::cout<<"\n################################## FINISHED DoE ##################################\n";

		return 0;
	}



	if(problemType == "Minimization" || problemType == "Maximization"){


		std::cout<<"\n################################## STARTING Optimization ##################################\n";
		runOptimization();

		std::cout<<"\n################################## FINISHED Optimization ##################################\n";

		return 0;
	}





	std::cout<<"ERROR: PROBLEM_TYPE is unknown!\n";
	abort();



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


