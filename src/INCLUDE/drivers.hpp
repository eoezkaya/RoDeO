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
#ifndef DRIVERS_HPP
#define DRIVERS_HPP


#include<armadillo>
#include<vector>
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "optimization.hpp"
#include "configkey.hpp"
using namespace arma;



class RoDeODriver{

private:


	std::string configFileName;

	std::vector<std::string> availableSurrogateModels;


	ConfigKeyList configKeys;
	ConfigKeyList configKeysObjectiveFunction;
	ConfigKeyList configKeysConstraintFunction;


	std::vector<ConstraintDefinition> constraints;
	int numberOfConstraints = 0;

	ObjectiveFunctionDefinition objectiveFunction;

	bool checkifProblemTypeIsValid(std::string) const;
	void checkIfProblemTypeIsSetProperly(void) const;

	bool isProblemTypeOptimization(std::string) const;
	bool isProblemTypeMinimization(std::string) const;
	bool isProblemTypeMaximization(std::string) const;

	SURROGATE_MODEL getSurrogateModelID(string) const;



public:
	RoDeODriver();

	void setDisplayOn(void);
	void setDisplayOff(void);



	void checkIfProblemDimensionIsSet(void) const;
	void checkIfNumberOfTrainingAndTestSamplesAreProper(void) const;
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


	void determineProblemDimensionAndBoxConstraintsFromTrainingData(void);


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
	unsigned int getDimension(void) const;
	std::string getProblemType(void) const;
	std::string getProblemName(void) const;


	void setConfigFilename(std::string);

	int runDriver(void);

	Optimizer setOptimizationStudy(void);
	void setOptimizationFeatures(Optimizer &) const;

	void runOptimization(void);
	void runSurrogateModelTest(void);
	void runDoE(void);
	void generateDoESamples(void);



	bool checkIfRunIsNecessary(int idConstraint) const;

	bool ifDisplayIsOn(void) const;
	void displayMessage(std::string inputString) const;

};



#endif
