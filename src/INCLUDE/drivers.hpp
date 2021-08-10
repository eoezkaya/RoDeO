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
#ifndef DRIVERS_HPP
#define DRIVERS_HPP


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



class RoDeODriver{

private:


	std::string configFileName;

	std::vector<std::string> availableSurrogateModels;



	std::vector<ConfigKey> configKeys;
	std::vector<ConfigKey> configKeysObjectiveFunction;
	std::vector<ConfigKey> configKeysConstraintFunction;

	std::vector<ConstraintDefinition> constraints;
	int numberOfConstraints = 0;

	ObjectiveFunctionDefinition objectiveFunction;

	bool checkifProblemTypeIsValid(std::string) const;
	void checkIfProblemTypeIsSetProperly(void) const;







public:
	RoDeODriver();

	void setDisplayOn(void);
	void setDisplayOff(void);

	void assignKeywordValue(std::pair <std::string,std::string> input, std::vector<ConfigKey> keywordArray);
	void assignKeywordValue(std::pair <std::string,vec> input, std::vector<ConfigKey> keywordArray);
	void assignKeywordValue(std::pair <std::string,double> input, std::vector<ConfigKey> keywordArray);
	void assignKeywordValue(std::pair <std::string,int> input, std::vector<ConfigKey> keywordArray);



	void assignKeywordValueWithIndex(std::string, int );

	void assignConstraintKeywordValueWithIndex(std::string, int );
	void assignObjectiveKeywordValueWithIndex(std::string, int);

	void assignKeywordValue(std::string, int );
	void assignKeywordValue(std::string, double );
	void assignKeywordValue(std::string, std::string);
	void assignKeywordValue(std::string, vec);


	int searchKeywordInString(std::string s, const std::vector<ConfigKey> &) const;
	int searchConfigKeywordInString(std::string ) const;
	int searchObjectiveConfigKeywordInString(std::string) const;
	int searchConstraintConfigKeywordInString(std::string s) const;

	std::string removeKeywordFromString(std::string, std::string) const;


	void checkIfProblemDimensionIsSet(void) const;
	void checkIfNumberOfTrainingAndTestSamplesAreProper(void) const;
	void checkIfObjectiveFunctionIsSetProperly(void) const;
	void checkIfSurrogateModelTypeIsOK(void) const;

	void checkSettingsForSurrogateModelTest(void) const;
	void checkSettingsForDoE(void) const;
	void checkSettingsForOptimization(void) const;


	void checkIfObjectiveFunctionNameIsDefined(void) const;
	void checkIfProblemDimensionIsSetProperly(void) const;

	void checkIfBoxConstraintsAreSetPropertly(void) const;

	void checkIfNumberOfTrainingSamplesIsDefined(void) const;
	void checkIfNumberOfTestSamplesIsDefined(void) const;



	void checkIfConstraintsAreProperlyDefined(void) const;

	bool ifIsAGradientBasedMethod(std::string) const;

	bool ifFeatureIsOn(std::string) const;
	bool ifFeatureIsOff(std::string) const;

	void determineProblemDimensionAndBoxConstraintsFromTrainingData(void);

	void printKeywords(void) const;
	void printKeywordsConstraint(void) const;
	void printKeywordsObjective(void) const;




	ConfigKey getConfigKey(int) const;
	ConfigKey getConfigKeyConstraint(int i) const;
	ConfigKey getConfigKeyObjective(int i) const;



	ConfigKey getConfigKey(std::string) const;
	ConfigKey getConfigKeyConstraint(std::string) const;
	ConfigKey getConfigKeyObjective(std::string) const;


	bool ifConfigKeyIsSet(std::string) const;
	bool ifConfigKeyIsSetConstraint(std::string) const;
	bool ifConfigKeyIsSetObjective(std::string) const;





	void abortifConfigKeyIsNotSet(std::string) const;


	std::string getConfigKeyStringValue(std::string) const;
	std::string getConfigKeyStringValueConstraint(std::string) const;
	std::string getConfigKeyStringValueObjective(std::string) const;


	std::vector<std::string> getConfigKeyStringVectorValue(std::string) const;
	std::string getConfigKeyStringVectorValueAtIndex(std::string, unsigned int) const;

	int getConfigKeyIntValue(std::string) const;
	int getConfigKeyIntValueObjective(std::string key) const;

	double getConfigKeyDoubleValue(std::string) const;
	vec getConfigKeyDoubleVectorValue(std::string) const;


	void checkConsistencyOfConfigParams(void) const;
	void readConfigFile(void);



	void extractObjectiveFunctionDefinitionFromString(std::string);
	void extractConfigDefinitionsFromString(std::string);
	void extractConstraintDefinitionsFromString(std::string);

	ObjectiveFunction  setObjectiveFunction(void) const;
	ConstraintFunction setConstraint(ConstraintDefinition constraintDefinition) const;


	void parseConstraintDefinition(std::string);
	void parseObjectiveFunctionDefinition(std::string);
	void printAllConstraintDefinitions(void) const;
	void printObjectiveFunctionDefinition(void) const;
	ObjectiveFunctionDefinition getObjectiveFunctionDefinition(void) const;
	ConstraintDefinition getConstraintDefinition(unsigned int) const;

	void setConfigFilename(std::string);

	int runDriver(void);

	Optimizer setOptimizationStudy(void);
	void setOptimizationFeatures(Optimizer &) const;

	void runOptimization(void);
	void runSurrogateModelTest(void);
	void runDoE(void);


	bool checkIfRunIsNecessary(int idConstraint) const;

	void displayMessage(std::string inputString) const;

};



#endif
