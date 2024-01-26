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
	FRIEND_TEST(OptimizationTest, setOptimizationHistory);
	FRIEND_TEST(OptimizationTest, updateOptimizationHistory);
	FRIEND_TEST(OptimizationTest, estimateConstraints);
	FRIEND_TEST(OptimizationTest, calculateFeasibilityProbabilities);
	FRIEND_TEST(OptimizationTest, findTheMostPromisingDesignGradientStep);
	FRIEND_TEST(OptimizationTest, setGradientGlobalOptimum);
	FRIEND_TEST(OptimizationTest, changeSettingsForAGradientBasedStep);
	FRIEND_TEST(OptimizationTest, determineMaxStepSizeForGradientStep);
	FRIEND_TEST(OptimizationTest, checkIfDesignTouchesBounds);
	FRIEND_TEST(OptimizationTest, checkIfDesignIsWithinBounds);
	FRIEND_TEST(OptimizationTest, doesObjectiveFunctionHaveGradients);


#endif

private:


	vec lowerBounds;
	vec upperBounds;

	vec lowerBoundsForAcqusitionFunctionMaximization;
	vec upperBoundsForAcqusitionFunctionMaximization;

	vec lowerBoundsForAcqusitionFunctionMaximizationGradientStep;
	vec upperBoundsForAcqusitionFunctionMaximizationGradientStep;



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

	unsigned int iterationNumberGradientStep = 0;
	unsigned int maximumIterationGradientStep = 5;
	double trustRegionFactorGradientStep = 1.0;
	bool WillGradientStepBePerformed = false;

	Design globalOptimalDesign;
	Design currentBestDesign;

	double initialImprovementValue = 0.0;


	double factorForGradientStepWindow = 0.01;


	unsigned int iterMaxAcquisitionFunction;
	unsigned int outerIterationNumber = 0;


	double sigmaFactor = 1.0;


	unsigned int numberOfDisceteVariables = 0;
	std::vector<double> incrementsForDiscreteVariables;
	std::vector<int> indicesForDiscreteVariables;


	unsigned int numberOfThreads = 1;

	void initializeOptimizerSettings(void);

	void setOptimizationHistoryConstraints(mat inputObjectiveFunction);
	void setOptimizationHistoryFeasibilityValues(void);
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
	bool doesObjectiveFunctionHaveGradients(void) const;


	void modifyBoundsForInnerIterations(void);

	bool isToZoomInIteration(unsigned int) const;

	void trainSurrogatesForConstraints();

	void setOptimizationHistory(void);
	void updateOptimizationHistory(Design d);
	void clearOptimizationHistoryFile(void) const;
	void prepareOptimizationHistoryFile(void) const;

	void estimateConstraints(DesignForBayesianOptimization &) const;

	double determineMaxStepSizeForGradientStep(rowvec x0, rowvec gradient) const;
	bool checkIfDesignTouchesBounds(const rowvec &x0) const;
	bool checkIfDesignIsWithinBounds(const rowvec &x0) const;

	void changeSettingsForAGradientBasedStep(void);
	void findTheMostPromisingDesignGradientStep(void);

	bool checkIfGlobalOptimaHasGradientVector(void) const;
	void setGradientGlobalOptimum(void);
	void setDataAddModeForGradientBasedStep(const Design &currentBestDesign);
	void findTheMostPromisingDesignToBeSimulated();
	void initializeCurrentBestDesign(void);
	void abortIfCurrentDesignHasANaN();
	void findPromisingDesignUnconstrainedGradientStep(
			DesignForBayesianOptimization &designToBeTried);
	bool checkIfBoxConstraintsAreSatisfied(const rowvec &dv) const;
	void decideIfAGradientStepShouldBeTakenForTheFirstIteration();
	double trimVectorSoThatItStaysWithinTheBounds(const arma::rowvec &x);

public:

	std::string name;

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maxNumberOfSamples = 0;

	unsigned int maxNumberOfSamplesLowFidelity = 0;

	unsigned int howOftenTrainModels = 100000;

	unsigned int sampleDim;

	unsigned int iterGradientEILoop = 100;

	bool ifBoxConstraintsSet = false;
	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;



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



	void setNumberOfThreads(unsigned int);


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
	void findTheMostPromisingDesignEGO(void);
	DesignForBayesianOptimization getDesignWithMaxExpectedImprovement(void) const;


};



#endif
