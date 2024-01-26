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

#ifndef DRIVERS_HPP
#define DRIVERS_HPP


#include<armadillo>
#include<vector>
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../../Optimizers/INCLUDE/optimization.hpp"
#include "../../SurrogateModels/INCLUDE/surrogate_model_tester.hpp"
#include "./configkey.hpp"
using namespace arma;

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif


class RoDeODriver{


#ifdef UNIT_TESTS
	friend class DriverTest;
	FRIEND_TEST(DriverTest, constructor);
	FRIEND_TEST(DriverTest, extractConfigDefinitionsFromString);
	FRIEND_TEST(DriverTest, extractObjectiveFunctionDefinitionFromString);
	FRIEND_TEST(DriverTest, extractConfigDefinitionFromString);
	FRIEND_TEST(DriverTest, extractConstraintDefinitionsFromString);

#endif


private:


	string configFileName;

	vector<string> availableSurrogateModels;


	ConfigKeyList configKeys;
	ConfigKeyList configKeysObjectiveFunction;
	ConfigKeyList configKeysConstraintFunction;


	vector<ObjectiveFunctionDefinition> constraintDefinitions;
	vector<ConstraintDefinition> constraintExpressions;
	int numberOfConstraints = 0;

	ObjectiveFunctionDefinition definitionObjectiveFunction;

	bool checkifProblemTypeIsValid(std::string) const;
	bool checkifProblemTypeIsOptimization(std::string s) const;
	bool checkifProblemTypeIsSurrogateTest(std::string s) const;


	void checkIfProblemTypeIsSetProperly(void) const;

	SURROGATE_MODEL getSurrogateModelID(string) const;

	string removeComments(const string &configText) const;

	void checkConsistencyOfObjectiveFunctionDefinition(void) const;
//	void checkConsistencyOfConstraint(ConstraintDefinition) const;
	void checkIfSurrogateModelTypeIsOK(void) const;
	void checkSettingsForSurrogateModelTest(void) const;
	void checkSettingsForOptimization(void) const;
	void checkIfObjectiveFunctionNameIsDefined(void) const;
	void checkIfProblemDimensionIsSetProperly(void) const;
	void checkIfBoxConstraintsAreSetPropertly(void) const;
	void checkConsistencyOfConfigParams(void) const;




	void abortIfModelTypeIsInvalid(const std::string &modelName) const;
	void addConfigKeysObjectiveFunction();
	void addConfigKeysConstraintFunctions();
	void addConfigKeysSurrogateModelTest();
	void addConfigKeysOptimization();
	void addConfigKeysGeneral();
	void addAvailableSurrogateModels();

	void extractObjectiveFunctionDefinitionFromString(std::string);
	void extractConfigDefinitionsFromString(std::string);
	void extractConstraintDefinitionsFromString(std::string);
	void setConstraintBoxConstraints(ConstraintFunction &constraintFunc) const;
	void checkIfSurrogateModelTypeIsOkMultiFidelity() const;
	void parseObjectiveFunctionDefinitionMultiFidelity();
	void parseConstraintDefinitionMultiFidelity(
			ObjectiveFunctionDefinition constraintFunctionDefinition);
	void abortIfProblemNameIsNotDefined(const std::string &name);
	void abortIfLowerOrUpperBoundsAreMissing(const arma::vec &lb,
			const arma::vec &ub);
	void addBoundsToOptimizationStudy(Optimizer &optimizationStudy);
	void addObjectiveFunctionToOptimizationStudy(Optimizer &optimizationStudy);
	void addConstraintsToOptimizationStudy(Optimizer &optimizationStudy);
	void addDiscreteParametersIfExistToOptimizationStudy(
			Optimizer &optimizationStudy);
	void abortSurrogateModelTestIfNecessaryParametersAreNotDefined();
	void addBoundsToSurrogateTester(SurrogateModelTester &surrogateTest);
	void addParametersToSurrogateTesterMultiFidelity(
			SurrogateModelTester &surrogateTest);
	void addModelTypeToSurrogateTester(SurrogateModelTester &surrogateTest);
	void addTrainingDataToSurrogateModelTester(
			SurrogateModelTester &surrogateTest);
	void addTestDataToSurrogateModelTester(SurrogateModelTester &surrogateTest);
	void addNumberOfTrainingIterationsToSurrogateTester(
			SurrogateModelTester &surrogateTest);
	void addLowFiModelTypeToSurrogateTester(
			SurrogateModelTester &surrogateTest);
	void addLowFiTrainingDataToSurrogateTester(
			SurrogateModelTester &surrogateTest);
	void addProblemNameToSurrogateTester(SurrogateModelTester &surrogateTest);
	void addDimensionToSurrogateTester(SurrogateModelTester &surrogateTest);
	void setOptimizationFeaturesMandatory(Optimizer &optimizationStudy) const;

	void setOptimizationFeaturesNumberOfThreads(
			Optimizer &optimizationStudy) const;

	OutputDevice output;



public:
	RoDeODriver();

	void setDisplayOn(void);
	void setDisplayOff(void);






	void readConfigFile(void);





	ObjectiveFunction  setObjectiveFunction(void) const;
	ConstraintFunction setConstraint(unsigned int) const;


	void parseConstraintDefinition(std::string);
	void parseObjectiveFunctionDefinition(std::string);
	void printAllConstraintDefinitions(void) const;
	void printObjectiveFunctionDefinition(void) const;

	ObjectiveFunctionDefinition getObjectiveFunctionDefinition(void) const;
	ObjectiveFunctionDefinition getConstraintDefinition(unsigned int) const;
	ConstraintDefinition getConstraintExpression(unsigned int) const;


	unsigned int getDimension(void) const;
	std::string getProblemType(void) const;
	std::string getProblemName(void) const;


	void setConfigFilename(std::string);

	void run(void);

	Optimizer setOptimizationStudy(void);
	void setOptimizationFeatures(Optimizer &) const;

	void runOptimization(void);
	void runSurrogateModelTest(void);


};



#endif
