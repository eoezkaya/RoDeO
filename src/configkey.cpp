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
#include "configkey.hpp"
#include "auxiliary_functions.hpp"
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


bool ConfigKeyList::ifKeyIsAlreadyIntheList(ConfigKey keyToAdd) const{

	std::string keyword = keyToAdd.name;

	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == keyword){

			return true;

		}

	}

	return false;
}

void ConfigKeyList::add(ConfigKey keyToAdd){

	assert(ifKeyIsAlreadyIntheList(keyToAdd) == false);

	keywordList.push_back(keyToAdd);
	numberOfKeys++;

}

unsigned int ConfigKeyList::countNumberOfElements(void) const{

	return numberOfKeys;

}



void ConfigKeyList::assignKeywordValue(std::pair <std::string,std::string> input) {


	std::string key   = input.first;
	std::string value = input.second;

	assert(!key.empty());
	assert(!value.empty());


	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == key){


			it->setValue(value);

		}

	}

}


void ConfigKeyList::assignKeywordValue(std::string key, std::string value) {

	assert(!key.empty());
	assert(!value.empty());


	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == key){


			it->setValue(value);

		}

	}

}

void ConfigKeyList::assignKeywordValue(std::string key, int value) {

	assert(!key.empty());



	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == key){


			it->setValue(value);

		}

	}

}



void ConfigKeyList::assignKeywordValue(std::string key, vec values) {


	assert(!key.empty());
	assert(values.size() > 0);

	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == key){

			it->setValue(values);

		}

	}

}


void ConfigKeyList::assignKeywordValueWithIndex(std::string s, int indxKeyword){

	assert(!s.empty());
	assert(indxKeyword >= 0);

	ConfigKey &keyWordToBeSet = keywordList.at(indxKeyword);

	keyWordToBeSet.setValue(s);


}



ConfigKey ConfigKeyList::getConfigKey(unsigned int indx) const{

	assert(indx<numberOfKeys);
	return this->keywordList.at(indx);

}


ConfigKey ConfigKeyList::getConfigKey(std::string key) const{

	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		if(it->name == key){

			return(*it);

		}

	}

	std::cout<<"ERROR: Invalid ConfigKey "<<key<<" \n";
	abort();


}

std::string ConfigKeyList::getConfigKeyStringValue(std::string key) const{

	std::string emptyString;

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	assert(keyFound.type == "string");

	if(keyFound.ifValueSet){

		return keyFound.stringValue;
	}
	else{

		return emptyString;
	}


}

int ConfigKeyList::getConfigKeyIntValue(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	assert(keyFound.type == "int");

	if(keyFound.ifValueSet){

		return keyFound.intValue;
	}
	else{

		return 0;
	}


}

double ConfigKeyList::getConfigKeyDoubleValue(std::string key) const{

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	assert(keyFound.type == "double");

	if(keyFound.ifValueSet){

		return keyFound.doubleValue;
	}
	else{

		return 0.0;
	}


}

std::vector<std::string> ConfigKeyList::getConfigKeyVectorStringValue(std::string key) const{

	std::vector<std::string> emptyVector;

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	assert(keyFound.type == "stringVector");

	if(keyFound.ifValueSet){

		return keyFound.vectorStringValue;
	}
	else{

		return emptyVector;
	}




}

vec ConfigKeyList::getConfigKeyVectorDoubleValue(std::string key) const{

	vec emptyVector;

	assert(!key.empty());
	ConfigKey keyFound = getConfigKey(key);

	assert(keyFound.type == "doubleVector");

	if(keyFound.ifValueSet){

		return keyFound.vectorDoubleValue;
	}
	else{

		return emptyVector;
	}




}


std::string ConfigKeyList::getConfigKeyStringVectorValueAtIndex(std::string key, unsigned int indx) const{


	assert(!key.empty());

	std::string emptyString;

	ConfigKey keyFound = getConfigKey(key);
	assert(keyFound.type == "stringVector");

	if(indx>=keyFound.vectorStringValue.size()){

		return emptyString;

	}


	if(keyFound.ifValueSet){

		return keyFound.vectorStringValue[indx];
	}
	else{

		return emptyString;
	}


}


void ConfigKeyList::printKeywords(void) const{

	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

		it->print();

	}




}

void ConfigKeyList::abortifConfigKeyIsNotSet(std::string key) const{

	ConfigKey keyword = getConfigKey(key);

	keyword.abortIfNotSet();


}

bool ConfigKeyList::ifConfigKeyIsSet(std::string key) const{

	ConfigKey keyword = getConfigKey(key);


	return keyword.ifValueSet;


}

int ConfigKeyList::searchKeywordInString(std::string s) const{

	s = removeSpacesFromString(s);

#if 0
	std::cout<<"Searching in string = "<<s<<"\n";
#endif

	int indx = 0;
	for(auto it = std::begin(keywordList); it != std::end(keywordList); ++it) {

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


				if(s.length() > found+keyToFound.length()){

					if ( s.at(found+keyToFound.length()) == '=' ||  s.at(found+keyToFound.length()) == ':'  ){

						return indx;

					}

				}

			}


		}

		indx++;
	}

#if 0
	std::cout<<"searchKeywordInString is done...\n";
#endif


	return -1;

}

bool ConfigKeyList::ifFeatureIsOn(std::string feature) const{

	assert(!feature.empty());

	std::string value = this->getConfigKeyStringValue(feature);

	return checkIfOn(value);


}

bool ConfigKeyList::ifFeatureIsOff(std::string feature) const{

	assert(!feature.empty());
	std::string value = this->getConfigKeyStringValue(feature);

	return checkIfOff(value);


}

void ConfigKeyList::parseString(std::string inputString){

	assert(inputString.empty()== false);

	std::stringstream iss(inputString);

	while(iss.good())
	{
		std::string singleLine;
		getline(iss,singleLine,'\n');

		singleLine = removeSpacesFromString(singleLine);
		int indxKeyword = searchKeywordInString(singleLine);


		if(indxKeyword != -1){

			ConfigKey temp = getConfigKey(indxKeyword);

			std::string keyword = temp.name;
			std::string cleanString;
			cleanString = removeKeywordFromString(singleLine, keyword);

			assignKeywordValueWithIndex(cleanString,indxKeyword);


		}
		else if(!singleLine.empty()){

			std::cout<<"ERROR: Cannot parse definition, something wrong with the input:\n";
			std::cout<<singleLine<<"\n";
			abort();
		}


	}
#if 0
	this->printKeywords();
#endif

}



