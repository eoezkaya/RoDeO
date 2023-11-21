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
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP


#include <armadillo>
#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../../Random/INCLUDE/random_functions.hpp"

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif


class Optimizer {

#ifdef UNIT_TESTS
	friend class OptimizationTest;
	FRIEND_TEST(OptimizationTest, constructor);
	FRIEND_TEST(OptimizationTest, setOptimizationProblem);
	FRIEND_TEST(OptimizationTest, reduceTrainingDataFiles);
	FRIEND_TEST(OptimizationTest, zoomInDesignSpace);
	FRIEND_TEST(OptimizationTest, setOptimizationHistory);
	FRIEND_TEST(OptimizationTest, updateOptimizationHistory);
	FRIEND_TEST(OptimizationTest, zoomInDesignSpace);
	FRIEND_TEST(OptimizationTest, estimateConstraints);
	FRIEND_TEST(OptimizationTest, calculateFeasibilityProbabilities);


#endif

private:


	vec lowerBounds;
	vec upperBounds;

	vec lowerBoundsForAcqusitionFunctionMaximization;
	vec upperBoundsForAcqusitionFunctionMaximization;


	std::string designVectorFileName;

	const std::string optimizationHistoryFileName = "optimizationHistory.csv";
	const std::string globalOptimumDesignFileName = "globalOptimumDesign";

	mat optimizationHistory;

	std::vector<Design> lowFidelityDesigns;
	std::vector<Design> highFidelityDesigns;


	std::vector<ConstraintFunction> constraintFunctions;
	ObjectiveFunction objFun;

	std::vector<DesignForBayesianOptimization> theMostPromisingDesigns;


	bool isHistoryFileInitialized = false;

	bool IfinitialValueForObjFunIsSet= false;

	Design globalOptimalDesign;

	double initialImprovementValue = 0.0;

	bool ifZoomInDesignSpaceIsAllowed = false;
	unsigned int howOftenZoomIn = 10;
	unsigned int minimumNumberOfSamplesAfterZoomIn = 0;


//	double zoomInFactor = 0.5;
//	double zoomFactorShrinkageRate = 0.75;
//	unsigned int howManySamplesReduceAfterZoomIn = 5;


	unsigned int iterMaxAcquisitionFunction;
	unsigned int outerIterationNumber = 0;


	double sigmaFactor = 1.0;
	double maximumSigma = 2.0;
	double minimumSigma = 0.1;


	unsigned int numberOfDisceteVariables = 0;
	std::vector<double> incrementsForDiscreteVariables;
	std::vector<int> indicesForDiscreteVariables;

	void setOptimizationHistoryConstraints(mat inputObjectiveFunction);
	void setOptimizationHistoryFeasibilityValues(mat inputObjectiveFunction);
	void calculateInitialImprovementValue(void);

	void findTheGlobalOptimalDesign(void);
	void initializeBoundsForAcquisitionFunctionMaximization();

	void modifySigmaFactor(void);

	void evaluateObjectiveFunction(Design &currentBestDesign);
	void evaluateObjectiveFunctionMultiFidelity(Design &currentBestDesign);
	void evaluateObjectiveFunctionSingleFidelity(Design &currentBestDesign);
	void evaluateObjectiveFunctionWithTangents(Design &currentBestDesign);
	void evaluateObjectFunctionWithAdjoints();
	void evaluateObjectFunctionWithPrimal();
	void evaluateObjectiveFunctionMultiFidelityWithBothPrimal(Design &currentBestDesign);
	void evaluateObjectiveFunctionMultiFidelityWithLowFiAdjoint(Design &currentBestDesign);

	void evaluateConstraints(Design &);

	bool areDiscreteParametersUsed(void) const;
	void roundDiscreteParameters(rowvec &);


	bool isConstrained(void) const;
	bool isNotConstrained(void) const;

	void zoomInDesignSpace(void);
	void reduceTrainingDataFiles(void) const;
	void reduceBoxConstraints(void);
	bool isToZoomInIteration(unsigned int) const;

	void trainSurrogatesForConstraints();

	void setOptimizationHistory(void);
	void updateOptimizationHistory(Design d);
	void clearOptimizationHistoryFile(void) const;
	void prepareOptimizationHistoryFile(void) const;

	void estimateConstraints(DesignForBayesianOptimization &) const;

public:

	std::string name;

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maxNumberOfSamples = 0;

	unsigned int maxNumberOfSamplesLowFidelity = 0;

	unsigned int howOftenTrainModels = 100000;

	double minDeltaXForZoom;

	unsigned int sampleDim;

	unsigned int iterGradientEILoop = 100;

	bool ifBoxConstraintsSet = false;
	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;
	bool ifreduceTrainingDataZoomIn = false;
	bool ifAdaptSigmaFactor         = false;



	OutputDevice output;

	Optimizer();
	Optimizer(std::string ,int);


	void setDimension(unsigned int);
	void setName(std::string);


	void setParameterToDiscrete(unsigned int, double);
	bool checkSettings(void) const;
	void print(void) const;
	void printConstraints(void) const;

	void performEfficientGlobalOptimization(void);

	void initializeSurrogates(void);
	void trainSurrogates(void);

	void setMaximumNumberOfIterations(unsigned int );
	void setMaximumNumberOfIterationsLowFidelity(unsigned int);

	void setMaximumNumberOfInnerIterations(unsigned int);

	void setBoxConstraints(Bounds boxConstraints);

	void setFileNameDesignVector(std::string filename);

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setZoomInOn(void);
	void setZoomInOff(void);
	void setHowOftenZoomIn(unsigned int value);
	void setMinimumNumberOfSamplesAfterZoomIn(unsigned int nSamples);

//	void setZoomFactor(double value);


	void setMaxSigmaFactor(double value);
	void setMinSigmaFactor(double value);




	void setHowOftenTrainModels(unsigned int value);




	void setInitialImprovementValue(double);
	void calculateImprovementValue(Design &);

	void addConstraint(ConstraintFunction &);


	void checkIfSettingsAreOK(void) const;
	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec) const;

	void addObjectFunction(ObjectiveFunction &);


	void addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &) const;

	void computeConstraintsandPenaltyTerm(Design &);

	void addConstraintValuesToData(Design &d);

	void calculateFeasibilityProbabilities(DesignForBayesianOptimization &) const;


	rowvec calculateGradientOfAcqusitionFunction(DesignForBayesianOptimization &) const;
	DesignForBayesianOptimization MaximizeAcqusitionFunctionGradientBased(DesignForBayesianOptimization ) const;
	void findTheMostPromisingDesign(void);
	DesignForBayesianOptimization getDesignWithMaxExpectedImprovement(void) const;


};



#endif
