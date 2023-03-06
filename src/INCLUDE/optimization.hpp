/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), RPTU
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
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "random_functions.hpp"



class Optimizer {

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

	double zoomInFactor = 0.5;
	double zoomFactorShrinkageRate = 0.75;

	unsigned int iterMaxAcqusitionFunction;

	unsigned int numberOfDisceteVariables = 0;
	std::vector<double> incrementsForDiscreteVariables;
	std::vector<int> indicesForDiscreteVariables;

	void setOptimizationHistoryConstraints(mat inputObjectiveFunction);
	void setOptimizationHistoryFeasibilityValues(mat inputObjectiveFunction);
	void calculateInitialImprovementValue(void);

	void findTheGlobalOptimalDesign(void);
	void initializeBoundsForAcquisitionFunctionMaximization();

public:

	std::string name;

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maxNumberOfSamples = 0;

	unsigned int maxNumberOfSamplesLowFidelity = 0;

	unsigned int howOftenTrainModels = 10;
	unsigned int howOftenZoomIn = 10;

	unsigned int sampleDim;

	unsigned int iterGradientEILoop = 100;


	bool ifVisualize = false;
	bool ifDisplay = false;
	bool ifBoxConstraintsSet = false;
	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;
	bool ifZoomInDesignSpaceIsAllowed = false;
	bool ifreduceTrainingDataZoomIn = false;


	OutputDevice output;

	Optimizer();
	Optimizer(std::string ,int);


	void setDimension(unsigned int);
	void setName(std::string);


	void setParameterToDiscrete(unsigned int, double);

	void roundDiscreteParameters(rowvec &);

	bool checkSettings(void) const;
	void print(void) const;
	void printConstraints(void) const;
	void visualizeOptimizationHistory(void) const;

	void EfficientGlobalOptimization(void);
//	void EfficientGlobalOptimization2(void);

	void initializeSurrogates(void);
	void trainSurrogates(void);

	void performDoE(unsigned int howManySamples, DoE_METHOD methodID);
	void cleanDoEFiles(void) const;
	void setProblemType(std::string);
	void setMaximumNumberOfIterations(unsigned int );
	void setMaximumNumberOfIterationsLowFidelity(unsigned int);

	void setMaximumNumberOfIterationsForEIMaximization(unsigned int);

	void setBoxConstraints(std::string filename="BoxConstraints.csv");
	void setBoxConstraints(double lb, double ub);
	void setBoxConstraints(vec lb, vec ub);
	void setBoxConstraints(Bounds boxConstraints);

	void setFileNameDesignVector(std::string filename);

	void setDisplayOn(void);
	void setDisplayOff(void);
	void setZoomInOn(void);
	void setZoomInOff(void);

	void setHowOftenTrainModels(unsigned int value);
	void setHowOftenZoomIn(unsigned int value);


	void zoomInDesignSpace(void);

	void setInitialImprovementValue(double);
	void calculateImprovementValue(Design &);

	void addConstraint(ConstraintFunction &);

	void evaluateConstraints(Design &);
	void addConstraintValuesToDoEData(Design &) const;


	void estimateConstraints(DesignForBayesianOptimization &) const;

	void checkIfSettingsAreOK(void) const;
	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec) const;

	void addObjectFunction(ObjectiveFunction &);


	void addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &) const;

	void computeConstraintsandPenaltyTerm(Design &);



	void findTheGlobalOptimalDesignMultiFidelity(void);


	void setOptimizationHistory(void);
	void updateOptimizationHistory(Design d);
	void clearOptimizationHistoryFile(void) const;
	void prepareOptimizationHistoryFile(void) const;

	mat getOptimizationHistory(void) const;


	void addConstraintValuesToData(Design &d);

	void calculateFeasibilityProbabilities(DesignForBayesianOptimization &) const;


	rowvec calculateGradientOfAcqusitionFunction(DesignForBayesianOptimization &) const;
	DesignForBayesianOptimization MaximizeAcqusitionFunctionGradientBased(DesignForBayesianOptimization ) const;
	void findTheMostPromisingDesign(unsigned int howManyDesigns = 1);
	DesignForBayesianOptimization getDesignWithMaxExpectedImprovement(void) const;

	rowvec generateRandomRowVectorAroundASample(void);


	bool ifConstrained(void) const;

	void displayMessage(std::string) const;


};



#endif
