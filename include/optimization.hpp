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
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP


#include <armadillo>
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "random_functions.hpp"



class COptimizer {

private:

	vec lowerBounds;
	vec upperBounds;

	std::string designVectorFileName;
	const std::string optimizationHistoryFileName = "optimizationHistory.csv";

	mat optimizationHistory;

	std::vector<ConstraintFunction> constraintFunctions;
	ObjectiveFunction objFun;

	std::vector<CDesignExpectedImprovement> theMostPromisingDesigns;

	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;


public:

	std::string name;

	unsigned int dimension;
	unsigned int numberOfConstraints;
	unsigned int maxNumberOfSamples;
	unsigned int howOftenTrainModels;

	unsigned int sampleDim;

	unsigned int iterGradientEILoop;
	std::string optimizationType;
	bool ifVisualize;

	unsigned int iterMaxEILoop;

	bool ifBoxConstraintsSet;
	bool ifCleanDoeFiles;

	COptimizer(std::string ,int, std::string = "minimize");
	bool checkSettings(void) const;
	void print(void) const;
	void printConstraints(void) const;
	void visualizeOptimizationHistory(void) const;
	void EfficientGlobalOptimization(void);

	void initializeSurrogates(void);
	void trainSurrogates(void);

	void performDoE(unsigned int howManySamples, DoE_METHOD methodID);
	void cleanDoEFiles(void) const;
	void setProblemType(std::string);
	void setMaximumNumberOfIterations(unsigned int );
	void setBoxConstraints(std::string filename="BoxConstraints.csv");
	void setBoxConstraints(double lb, double ub);
	void setBoxConstraints(vec lb, vec ub);
	void setFileNameDesignVector(std::string filename);

	void addConstraint(ConstraintFunction &constFunc);

	void evaluateConstraints(Design &d);
	void addConstraintValuesToDoEData(Design &d) const;


	void estimateConstraints(CDesignExpectedImprovement &design) const;

	void checkIfSettingsAreOK(void) const;
	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec constraintValues) const;

	void addObjectFunction(ObjectiveFunction &objFunc);


	void addPenaltyToExpectedImprovementForConstraints(CDesignExpectedImprovement &designCalculated) const;

	void computeConstraintsandPenaltyTerm(Design &);

	void updateOptimizationHistory(Design d);
	void clearOptimizationHistoryFile(void) const;
	void prepareOptimizationHistoryFile(void) const;


	void addConstraintValuesToData(Design &d);


	rowvec calculateEIGradient(CDesignExpectedImprovement &currentDesign) const;
	CDesignExpectedImprovement MaximizeEIGradientBased(CDesignExpectedImprovement initialDesignVector) const;
	void findTheMostPromisingDesign(unsigned int howManyDesigns = 1);
	CDesignExpectedImprovement getDesignWithMaxExpectedImprovement(void) const;

	rowvec generateRandomRowVectorAroundASample(void);




};

void testOptimizationHimmelblau(void);
void testOptimizationHimmelblauExternalExe(void);
void testOptimizationWingweight(void);
void testOptimizationEggholder(void);

#endif
