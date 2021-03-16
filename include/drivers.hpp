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
using namespace arma;




class RoDeODriver{

private:

	std::string *keywords;
	unsigned int numberOfKeywords;

	std::string configFileName;
	std::string problemType;
	std::string problemName;


	std::vector<std::string> availableSurrogateModels;

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maximumNumberOfSimulations = 0;
	unsigned int maximumNumberDoESamples = 0;

	unsigned int numberOfTrainingSamples = 0;
	unsigned int numberOfTestSamples = 0;

	std::string fileNametrainingDataForSurrogateTest;
	std::string fileNametestDataForSurrogateTest;

	vec boxConstraintsUpperBounds;
	vec boxConstraintsLowerBounds;
	vec constraintValues;

	bool ifObjectiveFunctionNameIsSet;
	bool ifObjectiveFunctionOutputFileIsSet;
	bool ifConstraintFunctionOutputFileIsSet;
	bool ifNumberOfConstraintsSet;
	bool ifProblemDimensionSet;
	bool ifUpperBoundsSet;
	bool ifLowerBoundsSet;
	bool ifProblemTypeSet;
	bool ifmaximumNumberOfSimulationsSet;
	bool ifmaximumNumberOfDoESamplesSet;
	bool ifexecutablePathObjectiveFunctionSet;
	bool ifDesignVectorFileNameSet;

	bool ifDoEFilesShouldBeCleaned;
	bool ifSurrogateModelTypeSet;

	bool ifFileNameTrainingDataForSurrogateTestIsSet = false;
	bool ifFileNameTestDataForSurrogateTestIsSet = false;


	std::vector<bool> IsGradientsAvailable;
	std::vector<std::string> executablesWithGradient;

	bool ifWarmStart = false;
	bool ifVisualization = false;

	std::string objectiveFunctionName;
	std::vector<std::string> constraintNames;
	std::vector<std::string> constraintTypes;
	std::vector<std::string> executableNames;
	std::vector<std::string> executablePaths;
	std::string designVectorFilename;
	std::vector<std::string> executableOutputFiles;

	std::string surrogateModelType;

	bool checkifProblemTypeIsValid(std::string) const;
	void setObjectiveFunction(ObjectiveFunction & );
	void setConstraint(ConstraintFunction & constraintFunc, unsigned int indx);



public:
	RoDeODriver();
	void checkIfProblemDimensionIsSet(void) const;
	void checkIfBoundsAreSetPropertly(void) const;
	void checkIfNumberOfTrainingAndTestSamplesAreProper(void) const;
	void checkIfObjectiveFunctionIsSetProperly(void) const;
	void checkIFSurrogateModelTypeIsOK(void) const;
	void checkSettingsForSurrogateModelTest(void) const;

	bool ifIsAGradientBasedMethod(std::string) const;
	void determineProblemDimensionAndBoxConstraintsFromTrainingData(void);

	void checkConsistencyOfConfigParams(void) const;
	void readConfigFile(void);
	int runDriver(void);
	void runOptimization(void) const;
	void runSurrogateModelTest(void);


};



#endif
