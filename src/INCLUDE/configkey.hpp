/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#ifndef CONFIGKEY_HPP
#define CONFIGKEY_HPP


#include<armadillo>
#include<vector>
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "optimization.hpp"
using namespace arma;





class ConfigKey{

public:
	std::string name;
	std::string type;

	double doubleValue = 0.0;
	int intValue = 0;
	std::string stringValue;
	vec vectorDoubleValue;
	std::vector<std::string> vectorStringValue;
	bool ifValueSet = false;

	ConfigKey(std::string, std::string);
	void print(void) const;
	void abortIfNotSet(void) const;

	void setValue(std::string);
	void setValue(vec);
	void setValue(int);
	void setValue(double);


};

class ConfigKeyList{

private:

	std::vector<ConfigKey> keywordList;
	unsigned int numberOfKeys = 0;

public:

	void add(ConfigKey);
	bool ifKeyIsAlreadyIntheList(ConfigKey keyToAdd) const;

	unsigned int countNumberOfElements(void) const;


	void printKeywords(void) const;

	void assignKeywordValue(std::pair <std::string,std::string> input);
	void assignKeywordValue(std::string key, std::string value);
	void assignKeywordValue(std::string key, int value);
	void assignKeywordValue(std::string key, vec values);

	void assignKeywordValueWithIndex(std::string s, int indxKeyword);


	ConfigKey getConfigKey(unsigned int) const;
	ConfigKey getConfigKey(std::string key) const;


	std::string getConfigKeyStringValue(std::string) const;
	int getConfigKeyIntValue(std::string) const;
	double getConfigKeyDoubleValue(std::string) const;
	std::vector<std::string> getConfigKeyVectorStringValue(std::string) const;
	vec getConfigKeyVectorDoubleValue(std::string) const;
	std::string getConfigKeyStringVectorValueAtIndex(std::string, unsigned int) const;



	void abortifConfigKeyIsNotSet(std::string key) const;
	bool ifConfigKeyIsSet(std::string key) const;

	int searchKeywordInString(std::string s) const;

	bool ifFeatureIsOn(std::string feature) const;
	bool ifFeatureIsOff(std::string feature) const;


};


#endif
