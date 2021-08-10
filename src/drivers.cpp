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
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


ConfigKey::ConfigKey(std::string name, std::string type){

	this->name = name;
	this->type = type;


}


void ConfigKey::print(void) const{

	std::cout<<name;


	if(ifValueSet){

		std::cout<<" : ";
		if(type == "string"){

			std::cout<<stringValue<<"\n";

		}
		if(type == "double"){

			std::cout<<doubleValue<<"\n";

		}
		if(type == "int"){

			std::cout<<intValue<<"\n";

		}

		if(type == "doubleVector"){

			printVector(vectorDoubleValue);

		}
		if(type == "stringVector"){

			printVector(vectorStringValue);

		}

	}
	else{

		std::cout<<"\n";
	}
}

void ConfigKey::abortIfNotSet(void) const{


	if(!this->ifValueSet){

		std::cout<<"ERROR: "<<this->name<<" is not set! Check the configuration file\n";
		abort();

	}


}


void ConfigKey::setValue(std::string val){


	val = removeSpacesFromString(val);

	if(type == "string"){

		this->stringValue = val;
	}

	if(type == "stringVector"){

		vectorStringValue = getStringValuesFromString(val,',');
#if 0
		printVector(vectorStringValue);
#endif

	}

	if(type == "doubleVector"){


		vectorDoubleValue = getDoubleValuesFromString(val,',');



	}
	if(type == "int"){

		this->intValue = std::stoi(val);


	}

	if(type == "double"){

		this->doubleValue = std::stod(val);


	}



	this->ifValueSet = true;



}

void ConfigKey::setValue(vec values){


	assert(type == "doubleVector");
	assert(values.size()>0);


	vectorDoubleValue = values;


	this->ifValueSet = true;


}

void ConfigKey::setValue(int value){

	assert(type == "int");

	this->intValue = value;
	this->ifValueSet = true;


}

void ConfigKey::setValue(double value){

	assert(type == "double");

	this->doubleValue = value;
	this->ifValueSet = true;


}



RoDeODriver::RoDeODriver(){


	configKeysObjectiveFunction.push_back(ConfigKey("NAME","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("DESIGN_VECTOR_FILE","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("OUTPUT_FILE","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("PATH","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("GRADIENT","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("EXECUTABLE","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("MARKER","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("MARKER_FOR_GRADIENT","string") );
	configKeysObjectiveFunction.push_back(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );

	configKeysConstraintFunction.push_back(ConfigKey("DEFINITION","string") );
	configKeysConstraintFunction.push_back(ConfigKey("DESIGN_VECTOR_FILE","string") );
	configKeysConstraintFunction.push_back(ConfigKey("OUTPUT_FILE","string") );
	configKeysConstraintFunction.push_back(ConfigKey("PATH","string") );
	configKeysConstraintFunction.push_back(ConfigKey("GRADIENT","string") );
	configKeysConstraintFunction.push_back(ConfigKey("EXECUTABLE","string") );
	configKeysConstraintFunction.push_back(ConfigKey("MARKER","string") );
	configKeysConstraintFunction.push_back(ConfigKey("MARKER_FOR_GRADIENT","string") );
	configKeysConstraintFunction.push_back(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );

	configKeys.push_back(ConfigKey("PROBLEM_TYPE","string") );
	configKeys.push_back(ConfigKey("PROBLEM_NAME","string") );
	configKeys.push_back(ConfigKey("DIMENSION","int") );
	configKeys.push_back(ConfigKey("UPPER_BOUNDS","doubleVector") );
	configKeys.push_back(ConfigKey("LOWER_BOUNDS","doubleVector") );


	configKeys.push_back(ConfigKey("NUMBER_OF_TEST_SAMPLES","int") );
	configKeys.push_back(ConfigKey("NUMBER_OF_TRAINING_SAMPLES","int") );


	configKeys.push_back(ConfigKey("FILENAME_TRAINING_DATA","string") );
	configKeys.push_back(ConfigKey("FILENAME_TEST_DATA","string") );

	configKeys.push_back(ConfigKey("SURROGATE_MODEL","string") );

	configKeys.push_back(ConfigKey("NUMBER_OF_DOE_SAMPLES","int") );
	configKeys.push_back(ConfigKey("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS","int") );

	configKeys.push_back(ConfigKey("WARM_START","string") );
	configKeys.push_back(ConfigKey("DESIGN_VECTOR_FILENAME","string") );

	configKeys.push_back(ConfigKey("VISUALIZATION","string") );
	configKeys.push_back(ConfigKey("DISPLAY","string") );

	configKeys.push_back(ConfigKey("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION","int") );


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


	assignKeywordValue("DISPLAY",std::string("ON"));


}
void RoDeODriver::setDisplayOff(void){


	assignKeywordValue("DISPLAY",std::string("OFF"));
}



ConfigKey RoDeODriver::getConfigKey(int i) const{

	assert(i>= 0);

	return this->configKeys.at(i);

}

ConfigKey RoDeODriver::getConfigKeyConstraint(int i) const{

	assert(i>= 0);

	return this->configKeysConstraintFunction.at(i);


}


ConfigKey RoDeODriver::getConfigKeyObjective(int i) const{

	assert(i>= 0);

	return this->configKeysObjectiveFunction.at(i);

}




std::vector<std::string> RoDeODriver::getConfigKeyStringVectorValue(std::string key) const{

	assert(!key.empty());

	ConfigKey keyFound = getConfigKey(key);

	return keyFound.vectorStringValue;


}

std::string RoDeODriver::getConfigKeyStringVectorValueAtIndex(std::string key, unsigned int indx) const{

	assert(!key.empty());

	ConfigKey keyFound = getConfigKey(key);

	return keyFound.vectorStringValue[indx];


}




std::string RoDeODriver::getConfigKeyStringValue(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	return keyFound.stringValue;


}

std::string RoDeODriver::getConfigKeyStringValueConstraint(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKeyConstraint(key);

	return keyFound.stringValue;


}

std::string RoDeODriver::getConfigKeyStringValueObjective(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKeyObjective(key);

	return keyFound.stringValue;


}

int RoDeODriver::getConfigKeyIntValueObjective(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKeyObjective(key);

	return keyFound.intValue;


}


int RoDeODriver::getConfigKeyIntValue(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	return keyFound.intValue;



}


double RoDeODriver::getConfigKeyDoubleValue(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	return keyFound.doubleValue;


}

vec RoDeODriver::getConfigKeyDoubleVectorValue(std::string key) const{


	ConfigKey keyFound = getConfigKey(key);

	return keyFound.vectorDoubleValue;


}



ConfigKey RoDeODriver::getConfigKey(std::string key) const{

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		if(it->name == key){

			return(*it);

		}

	}

	std::cout<<"ERROR: Invalid ConfigKey "<<key<<" \n";
	abort();


}


ConfigKey RoDeODriver::getConfigKeyConstraint(std::string key) const{

	for(auto it = std::begin(configKeysConstraintFunction); it != std::end(configKeysConstraintFunction); ++it) {

		if(it->name == key){

			return(*it);

		}

	}

	std::cout<<"ERROR: Invalid ConfigKey for the constraint function "<<key<<" \n";
	abort();

}

ConfigKey RoDeODriver::getConfigKeyObjective(std::string key) const{


	for(auto it = std::begin(configKeysObjectiveFunction); it != std::end(configKeysObjectiveFunction); ++it) {

		if(it->name == key){

			return(*it);

		}

	}

	std::cout<<"ERROR: Invalid ConfigKey for the objective function "<<key<<" \n";
	abort();


}




bool RoDeODriver::ifConfigKeyIsSet(std::string key) const{

	ConfigKey keyword = getConfigKey(key);


	return keyword.ifValueSet;


}

bool RoDeODriver::ifConfigKeyIsSetConstraint(std::string key) const{

	ConfigKey keyword = getConfigKeyConstraint(key);


	return keyword.ifValueSet;


}


bool RoDeODriver::ifConfigKeyIsSetObjective(std::string key) const{

	ConfigKey keyword = getConfigKeyObjective(key);


	return keyword.ifValueSet;


}



void RoDeODriver::abortifConfigKeyIsNotSet(std::string key) const{

	ConfigKey keyword = getConfigKey(key);

	keyword.abortIfNotSet();


}




void RoDeODriver::printKeywords(void) const{

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		it->print();

	}




}


void RoDeODriver::printKeywordsConstraint(void) const{

	for(auto it = std::begin(configKeysConstraintFunction); it != std::end(configKeysConstraintFunction); ++it) {

		it->print();

	}




}

void RoDeODriver::printKeywordsObjective(void) const{


	for(auto it = std::begin(configKeysObjectiveFunction); it != std::end(configKeysObjectiveFunction); ++it) {

		it->print();

	}




}




void RoDeODriver::assignKeywordValue(std::pair <std::string,std::string> input, std::vector<ConfigKey> keywordArray) {


	std::string key   = input.first;
	std::string value = input.second;

	assert(!key.empty());
	assert(!value.empty());


	for(auto it = std::begin(keywordArray); it != std::end(keywordArray); ++it) {

		if(it->name == key){


			it->setValue(value);

		}

	}

}





void RoDeODriver::assignKeywordValue(std::string key, std::string value) {

	assert(!key.empty());
	assert(!value.empty());


	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		if(it->name == key){


			it->setValue(value);

		}

	}

}

void RoDeODriver::assignKeywordValue(std::pair <std::string, vec> input, std::vector<ConfigKey> keywordArray) {


	std::string key   = input.first;
	vec values = input.second;


	assert(!key.empty());
	assert(values.size() > 0);

	for(auto it = std::begin(keywordArray); it != std::end(keywordArray); ++it) {

		if(it->name == key){

			it->setValue(values);

		}

	}

}


void RoDeODriver::assignKeywordValue(std::string key, vec values) {


	assert(!key.empty());
	assert(values.size() > 0);

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		if(it->name == key){

			it->setValue(values);

		}

	}

}



void RoDeODriver::assignKeywordValue(std::pair <std::string, int> input, std::vector<ConfigKey> keywordArray) {

	std::string key   = input.first;
	int value = input.second;

	assert(!key.empty());

	for(auto it = std::begin(keywordArray); it != std::end(keywordArray); ++it) {

		if(it->name == key){

			it->setValue(value);

		}

	}

}





void RoDeODriver::assignKeywordValue(std::string key, int value) {

	assert(!key.empty());

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {



		if(it->name == key){



			it->setValue(value);

		}

	}

}


void RoDeODriver::assignKeywordValue(std::pair <std::string, double> input, std::vector<ConfigKey> keywordArray) {

	std::string key   = input.first;
	double value = input.second;


	assert(!key.empty());

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		if(it->name == key){

			it->setValue(value);

		}

	}

}




void RoDeODriver::assignKeywordValue(std::string key, double value) {

	assert(!key.empty());

	for(auto it = std::begin(configKeys); it != std::end(configKeys); ++it) {

		if(it->name == key){

			it->setValue(value);

		}

	}

}


void RoDeODriver::assignKeywordValueWithIndex(std::string s, int indxKeyword){

	assert(!s.empty());
	assert(indxKeyword >= 0);

	ConfigKey &keyWordToBeSet = this->configKeys.at(indxKeyword);

	keyWordToBeSet.setValue(s);


}


void RoDeODriver::assignConstraintKeywordValueWithIndex(std::string s, int indxKeyword){

	assert(!s.empty());
	assert(indxKeyword >= 0);

	ConfigKey &keyWordToBeSet = this->configKeysConstraintFunction.at(indxKeyword);

	keyWordToBeSet.setValue(s);


}

void RoDeODriver::assignObjectiveKeywordValueWithIndex(std::string s, int indxKeyword){

	assert(!s.empty());
	assert(indxKeyword >= 0);

	ConfigKey &keyWordToBeSet = this->configKeysObjectiveFunction.at(indxKeyword);

	keyWordToBeSet.setValue(s);


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



int RoDeODriver::searchKeywordInString(std::string s, const std::vector<ConfigKey> &keywordArray) const{


#if 0
	std::cout<<"Searching in string = "<<s<<"\n";
#endif

	int indx = 0;
	for(auto it = std::begin(keywordArray); it != std::end(keywordArray); ++it) {

		std::string keyToFound = it->name;

#if 0
		std::cout<<"keyToFound = "<<keyToFound<<"\n";
#endif

		std::size_t found = s.find(keyToFound);

		if (found!=std::string::npos){

			if(found == 0){

#if 0
				std::cout<<"key Found\n";
#endif

				if ( s.at(found+keyToFound.length()) == '=' ||  s.at(found+keyToFound.length()) == ':'  ){

					return indx;

				}

			}


		}

		indx++;
	}

	return -1;




}


int RoDeODriver::searchConfigKeywordInString(std::string s) const{


	return searchKeywordInString(s,configKeys);


}


int RoDeODriver::searchObjectiveConfigKeywordInString(std::string s) const{


	return searchKeywordInString(s,configKeysObjectiveFunction);

}


int RoDeODriver::searchConstraintConfigKeywordInString(std::string s) const{


	return searchKeywordInString(s,configKeysConstraintFunction);


}




void RoDeODriver::readConfigFile(void){


	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Reading the configuration file : "<< configFileName <<"\n";

	}

	std::string stringCompleteFile;

	readFileToaString(configFileName, stringCompleteFile);

	if(ifFeatureIsOn("DISPLAY")){

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

	ConfigKey configKeyName = getConfigKeyObjective("NAME");
	configKeyName.abortIfNotSet();


}


void RoDeODriver::checkIfProblemDimensionIsSetProperly(void) const{

	abortifConfigKeyIsNotSet("DIMENSION");

	ConfigKey dimension = this->getConfigKey("DIMENSION");


	if(dimension.intValue > 1000){

		std::cout<<"ERROR: Problem dimension is too large, did you set DIMENSION properly?\n";
		abort();

	}

	if(dimension.intValue <= 0){

		std::cout<<"ERROR: Problem dimension must be a positive integer, did you set DIMENSION properly?\n";
		abort();

	}



}


void RoDeODriver::checkIfObjectiveFunctionIsSetProperly(void) const{

	ConfigKey configKeyName = getConfigKeyObjective("NAME");
	configKeyName.abortIfNotSet();


}




void RoDeODriver::checkIfBoxConstraintsAreSetPropertly(void) const{

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Checking box constraint settings...\n";
	}

	ConfigKey ub = this->getConfigKey("UPPER_BOUNDS");
	ConfigKey lb = this->getConfigKey("LOWER_BOUNDS");
	ConfigKey dimension = this->getConfigKey("DIMENSION");
	ub.abortIfNotSet();
	lb.abortIfNotSet();

	int sizeUb = ub.vectorDoubleValue.size();
	int sizeLb = lb.vectorDoubleValue.size();

	if(sizeUb != sizeLb || sizeUb != dimension.intValue){

		std::cout<<"ERROR: Dimension of bounds does not match with the problem dimension, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?\n";
		abort();

	}

	for(int i=0; i<dimension.intValue; i++){

		if( ub.vectorDoubleValue(i) <= lb.vectorDoubleValue(i)){

			std::cout<<"ERROR: Lower bounds cannot be greater or equal than upper bounds, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?\n";
			abort();


		}


	}


}






void RoDeODriver::checkSettingsForSurrogateModelTest(void) const{


	checkIfSurrogateModelTypeIsOK();



	if(!ifConfigKeyIsSet("FILENAME_TRAINING_DATA") || !ifConfigKeyIsSet("FILENAME_TEST_DATA") ){

		checkIfProblemDimensionIsSetProperly();

		checkIfBoxConstraintsAreSetPropertly();

		checkIfObjectiveFunctionIsSetProperly();


	}


	if(!ifConfigKeyIsSet("FILENAME_TRAINING_DATA")) {

		checkIfNumberOfTrainingSamplesIsDefined();

	}

	if(!ifConfigKeyIsSet("FILENAME_TEST_DATA")) {

		checkIfNumberOfTestSamplesIsDefined();

	}




}



void RoDeODriver::checkIfNumberOfTrainingSamplesIsDefined(void) const{

	abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");

}

void RoDeODriver::checkIfNumberOfTestSamplesIsDefined(void) const{

	abortifConfigKeyIsNotSet("NUMBER_OF_TRAINING_SAMPLES");

}




void RoDeODriver::checkIfSurrogateModelTypeIsOK(void) const{

	ConfigKey surrogateModelType = this->getConfigKey("SURROGATE_MODEL");

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


	if(ifFeatureIsOn("DISPLAY")){

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
	checkIfObjectiveFunctionIsSetProperly();
	checkIfConstraintsAreProperlyDefined();

	abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");


}

void RoDeODriver::checkSettingsForOptimization(void) const{

	checkIfProblemDimensionIsSetProperly();
	checkIfBoxConstraintsAreSetPropertly();
	checkIfObjectiveFunctionIsSetProperly();
	checkIfConstraintsAreProperlyDefined();


	abortifConfigKeyIsNotSet("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

}





void RoDeODriver::checkConsistencyOfConfigParams(void) const{

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Checking consistency of the configuration parameters...\n";
	}


	checkIfProblemTypeIsSetProperly();

	checkIfObjectiveFunctionNameIsDefined();


	std::string type = getConfigKeyStringValue("PROBLEM_TYPE");


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


	ConfigKey problemType = this->getConfigKey("PROBLEM_TYPE");
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

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Setting objective function...\n";

	}


	std::string objFunName = this->getConfigKeyStringValueObjective("NAME");
	int dim = this->getConfigKeyIntValue("DIMENSION");
	ObjectiveFunction objFunc(objFunName, dim);


	objFunc.setParametersByDefinition(this->objectiveFunction);


	vec lb = this->getConfigKeyDoubleVectorValue("LOWER_BOUNDS");
	vec ub = this->getConfigKeyDoubleVectorValue("UPPER_BOUNDS");

	objFunc.setParameterBounds(lb,ub);


	std::string ifGradientAvailable = this->getConfigKeyStringValueObjective("GRADIENT");


	if(checkIfOn(ifGradientAvailable)){

		objFunc.setGradientOn();

	}


	unsigned int nIterForSurrogateTraining = 10000;

	if(this->ifConfigKeyIsSetObjective("NUMBER_OF_TRAINING_ITERATIONS")){

		nIterForSurrogateTraining = this->getConfigKeyIntValueObjective("NUMBER_OF_TRAINING_ITERATIONS");

	}


	objFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);

	if(ifFeatureIsOn("DISPLAY")){

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
		int indxKeyword = searchConstraintConfigKeywordInString(singleLine);


		if(indxKeyword != -1){
			std::string keyword = this->configKeysConstraintFunction[indxKeyword].name;
			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			assignConstraintKeywordValueWithIndex(cleanString,indxKeyword);


		}


	}

#if 0
	printKeywordsConstraint();
#endif



	definitionBuffer = this->getConfigKeyStringValueConstraint("DEFINITION");
	designVectorFilename = this->getConfigKeyStringValueConstraint("DESIGN_VECTOR_FILE");
	executableName = this->getConfigKeyStringValueConstraint("EXECUTABLE");
	outputFilename = this->getConfigKeyStringValueConstraint("OUTPUT_FILE");
	exePath = this->getConfigKeyStringValueConstraint("PATH");
	marker = this->getConfigKeyStringValueConstraint("MARKER");
	markerGradient = this->getConfigKeyStringValueConstraint("MARKER_FOR_GRADIENT");
	ifGradient = this->getConfigKeyStringValueConstraint("GRADIENT");

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

	if(ifFeatureIsOn("DISPLAY")){

		std::cout <<"Adding a constraint definition with ID = "<< result.ID<<"\n";
	}


	this->constraints.push_back(result);


}


ObjectiveFunctionDefinition RoDeODriver::getObjectiveFunctionDefinition(void) const{

	return this->objectiveFunction;


}


ConstraintDefinition RoDeODriver::getConstraintDefinition(unsigned int i) const{

	assert(i<constraints.size());
	return this->constraints.at(i);


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

	while(iss.good())
	{
		std::string singleLine;
		getline(iss,singleLine,'\n');

		singleLine = removeSpacesFromString(singleLine);
		int indxKeyword = searchObjectiveConfigKeywordInString(singleLine);


		if(indxKeyword != -1){
			std::string keyword = this->configKeysObjectiveFunction[indxKeyword].name;


			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			assignObjectiveKeywordValueWithIndex(cleanString,indxKeyword);


		}


	}

#if 0
	printKeywordsObjective();
#endif



	name = this->getConfigKeyStringValueObjective("NAME");
	designVectorFilename = this->getConfigKeyStringValueObjective("DESIGN_VECTOR_FILE");
	executableName = this->getConfigKeyStringValueObjective("EXECUTABLE");
	outputFilename = this->getConfigKeyStringValueObjective("OUTPUT_FILE");
	exePath = this->getConfigKeyStringValueObjective("PATH");
	marker = this->getConfigKeyStringValueObjective("MARKER");


	markerGradient = this->getConfigKeyStringValueObjective("MARKER_FOR_GRADIENT");
	gradient = this->getConfigKeyStringValueObjective("GRADIENT");



	this->objectiveFunction.name = name;
	this->objectiveFunction.executableName = executableName;
	this->objectiveFunction.outputFilename = outputFilename;
	this->objectiveFunction.designVectorFilename =  designVectorFilename;
	this->objectiveFunction.path = exePath;
	this->objectiveFunction.marker = marker;
	this->objectiveFunction.markerForGradient = markerGradient;

	if(checkIfOn(gradient)){

		this->objectiveFunction.ifGradient = true;

	}

	objectiveFunction.ifDefined = true;

#if 0
	this->objectiveFunction.print();
#endif


}



void RoDeODriver::extractObjectiveFunctionDefinitionFromString(std::string inputString){

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Extracting objective function definition\n";

	}

	std::size_t foundObjectiveFunction = inputString.find("OBJECTIVE_FUNCTION");
	if (foundObjectiveFunction != std::string::npos){

		if(ifFeatureIsOn("DISPLAY")){

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
		int indxKeyword = searchConfigKeywordInString(singleLine);


		if(indxKeyword != -1){


			std::string keyword = this->configKeys[indxKeyword].name;
			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			assignKeywordValueWithIndex(cleanString,indxKeyword);


		}


	}


}



ConstraintFunction RoDeODriver::setConstraint(ConstraintDefinition constraintDefinition) const{

	assert(constraintDefinition.ID >= 0);

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Setting the constraint with ID = "<< constraintDefinition.ID << "\n";

	}


	int dim = this->getConfigKeyIntValue("DIMENSION");


	ConstraintFunction constraintFunc(constraintDefinition.name, dim);
	constraintFunc.setParametersByDefinition(constraintDefinition);

	vec lb = this->getConfigKeyDoubleVectorValue("LOWER_BOUNDS");
	vec ub = this->getConfigKeyDoubleVectorValue("UPPER_BOUNDS");
	constraintFunc.setParameterBounds(lb,ub);


	if(constraintDefinition.ifGradient){

		constraintFunc.setGradientOn();

	}


	unsigned int nIterForSurrogateTraining = 10000;

	if(this->ifConfigKeyIsSetConstraint("NUMBER_OF_TRAINING_ITERATIONS")){

		nIterForSurrogateTraining = this->getConfigKeyIntValueObjective("NUMBER_OF_TRAINING_ITERATIONS");

	}
	constraintFunc.setNumberOfTrainingIterationsForSurrogateModel(nIterForSurrogateTraining);



	if(ifFeatureIsOn("DISPLAY")){


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

	if(ifFeatureIsOn("DISPLAY")){

		optimizationStudy.setDisplayOn();

	}

	abortifConfigKeyIsNotSet("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

	int nFunctionEvals = this->getConfigKeyIntValue("MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS");

	optimizationStudy.setMaximumNumberOfIterations(nFunctionEvals);


	int nIterationsForEIMaximization = this->getConfigKeyIntValue("DIMENSION")* 100000;

	if(ifConfigKeyIsSet("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION")){


		nIterationsForEIMaximization = this->getConfigKeyIntValue("NUMBER_OF_ITERATIONS_FOR_EXPECTED_IMPROVEMENT_MAXIMIZATION");

		optimizationStudy.setMaximumNumberOfIterationsForEIMaximization(nIterationsForEIMaximization);

	}



}

void RoDeODriver::runOptimization(void){

	displayMessage("Running Optimization...\n");

	Optimizer optimizationStudy = setOptimizationStudy();

	setOptimizationFeatures(optimizationStudy);


	std::string WarmStart = "OFF";

	if(ifConfigKeyIsSet("WARM_START")){


		WarmStart = getConfigKeyStringValue("WARM_START");

		std::string msg = "Warm start = " + WarmStart;
		displayMessage(msg);


	}

	if(ifFeatureIsOn("DISPLAY")){

		optimizationStudy.print();

	}



	if(checkIfOff(WarmStart)){

		abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");
		int maximumNumberDoESamples = this->getConfigKeyIntValue("NUMBER_OF_DOE_SAMPLES");
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

bool RoDeODriver::ifFeatureIsOn(std::string feature) const{

	std::string value = getConfigKeyStringValue(feature);

	return checkIfOn(value);


}
bool RoDeODriver::ifFeatureIsOff(std::string feature) const{

	std::string value = getConfigKeyStringValue(feature);

	return checkIfOff(value);


}


void RoDeODriver::determineProblemDimensionAndBoxConstraintsFromTrainingData(void){

	/* These two are absolutely required */
	this->abortifConfigKeyIsNotSet("FILENAME_TRAINING_DATA");
	this->abortifConfigKeyIsNotSet("SURROGATE_MODEL");


	std::string fileName = this->getConfigKeyStringValue("FILENAME_TRAINING_DATA");
	std::string surrogateModelType = this->getConfigKeyStringValue("SURROGATE_MODEL");



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


	assignKeywordValue("DIMENSION",dim);

	vec lb,ub;


	lb = zeros<vec>(dim);
	ub = zeros<vec>(dim);

	for(unsigned int i=0; i<dim; i++){

		vec columnOfData = bufferData.col(i);

		lb(i) = min(columnOfData);
		ub(i) = max(columnOfData);

	}


	assignKeywordValue("LOWER_BOUNDS",lb);
	assignKeywordValue("UPPER_BOUNDS",ub);


	if(ifFeatureIsOn("DISPLAY")){


		printVector(lb,"boxConstraintsLowerBounds");
		printVector(ub,"boxConstraintsUpperBounds");
		std::cout<<"dimension = "<<dim<<"\n";

	}

}

Optimizer RoDeODriver::setOptimizationStudy(void) {

	std::string name = this->getConfigKeyStringValue("PROBLEM_NAME");
	std::string type = this->getConfigKeyStringValue("PROBLEM_TYPE");
	int dim = this->getConfigKeyIntValue("DIMENSION");

	Optimizer optimizationStudy(name, dim, type);
	vec lb = this->getConfigKeyDoubleVectorValue("LOWER_BOUNDS");
	vec ub = this->getConfigKeyDoubleVectorValue("UPPER_BOUNDS");

	optimizationStudy.setBoxConstraints(lb,ub);

	std::string dvFilename = this->getConfigKeyStringValueObjective("DESIGN_VECTOR_FILE");

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


	if(ifFeatureIsOn("WARM_START")){

		ifWarmStart = true;

	}

	if(ifWarmStart == false){

		optimizationStudy.cleanDoEFiles();

	}


	abortifConfigKeyIsNotSet("NUMBER_OF_DOE_SAMPLES");
	int maximumNumberDoESamples = getConfigKeyIntValue("NUMBER_OF_DOE_SAMPLES");

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


	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<"Checking if a run is necessary for the constraint with ID = "<<idConstraint<<"\n";
	}

	for ( auto i = constraints.begin(); i != constraints.end(); i++ ) {

		if(i->executableName == exeNameOftheConstraint && i->ID < idConstraint) return false;

	}


	return true;

}




void RoDeODriver::runSurrogateModelTest(void){

	if(ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		determineProblemDimensionAndBoxConstraintsFromTrainingData();

	}

	std::string surrogateModelType = this->getConfigKeyStringValue("SURROGATE_MODEL");
	std::string objectiveFunctionName = this->getConfigKeyStringValueObjective("NAME");

	int dimension = this->getConfigKeyIntValue("DIMENSION");

	TestFunction TestFunction(objectiveFunctionName, dimension);

	if(  ifIsAGradientBasedMethod(surrogateModelType)){

		TestFunction.setGradientsOn();

	}

	if(ifConfigKeyIsSetObjective("NUMBER_OF_TRAINING_ITERATIONS")){

		int nIter = this->getConfigKeyIntValueObjective("NUMBER_OF_TRAINING_ITERATIONS");
		assert(nIter>0);
		assert(nIter<1000000);
		TestFunction.setNumberOfTrainingIterations(nIter);
	}



	if(ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		std::string inputFilename = this->getConfigKeyStringValue("FILENAME_TRAINING_DATA");
		assert(!inputFilename.empty());
		TestFunction.setNameFilenameTrainingData(inputFilename);

	}
	else{

		std::string exeName = this->getConfigKeyStringValueObjective("EXECUTABLE");

		TestFunction.setNameOfExecutable(exeName);

		std::string designVectorFilename = this->getConfigKeyStringValueObjective("DESIGN_VECTOR_FILE");

		TestFunction.setNameOfInputForExecutable(designVectorFilename);

		std::string outputFilename = this->getConfigKeyStringValueObjective("OUTPUT_FILE");
		TestFunction.setNameOfOutputForExecutable(outputFilename);

		int numberOfTrainingSamples =  this->getConfigKeyIntValue("NUMBER_OF_TRAINING_SAMPLES");
		assert(numberOfTrainingSamples>0);
		TestFunction.setNumberOfTrainingSamples(numberOfTrainingSamples);



	}

	if(ifConfigKeyIsSet("FILENAME_TEST_DATA")){

		std::string inputFilename = this->getConfigKeyStringValue("FILENAME_TEST_DATA");
		assert(!inputFilename.empty());
		TestFunction.setNameFilenameTestData(inputFilename);


	}
	else{

		int numberOfTestSamples =  this->getConfigKeyIntValue("NUMBER_OF_TEST_SAMPLES");
		assert(numberOfTestSamples>0);
		TestFunction.setNumberOfTestSamples(numberOfTestSamples);

	}



	vec lb = this->getConfigKeyDoubleVectorValue("LOWER_BOUNDS");
	vec ub = this->getConfigKeyDoubleVectorValue("UPPER_BOUNDS");

	TestFunction.setBoxConstraints(lb, ub);


	if(ifConfigKeyIsSet("FILENAME_TRAINING_DATA")){

		TestFunction.readFileTrainingData();

	}
	else{

		TestFunction.generateSamplesInputTrainingData();
		TestFunction.generateTrainingSamples();

	}

	if(ifConfigKeyIsSet("FILENAME_TEST_DATA")){

		TestFunction.readFileTestData();

	}
	else{

		TestFunction.generateSamplesInputTestData();
		TestFunction.generateTestSamples();
	}



	if(ifConfigKeyIsSet("VISUALIZATION")) {

		std::string flag = this->getConfigKeyStringValue("VISUALIZATION");

		if(flag == "ON" || flag == "YES"){

			TestFunction.setVisualizationOn();
		}

	}


	if(ifConfigKeyIsSet("DISPLAY")) {

		std::string flag = this->getConfigKeyStringValue("DISPLAY");

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

	std::string problemType = this->getConfigKeyStringValue("PROBLEM_TYPE");

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

	if(ifFeatureIsOn("DISPLAY")){

		std::cout<<inputString<<"\n";


	}


}


