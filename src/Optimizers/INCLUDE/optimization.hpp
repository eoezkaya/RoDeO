/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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

#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../INCLUDE/globalOptimalDesign.hpp"
#include "../INCLUDE/optimization_history.hpp"
#include "../INCLUDE/optimization_logger.hpp"


namespace Rodop{

class OptimizationStatistics{

public:
	unsigned int numberOfObjectiveFunctionEvaluations = 0;
	unsigned int numberOfObjectiveGradientEvaluations = 0;

	std::vector<unsigned int> numberOfConstraintEvaluations;
	std::vector<unsigned int> numberOfConstraintGradientEvaluations;

	std::time_t startTime;
	std::time_t endTime;
	std::time_t startTimeIteration;
	std::time_t endTimeIteration;



	void getTime(std::time_t &time){
		auto now = std::chrono::system_clock::now();
		time = std::chrono::system_clock::to_time_t(now);
	}


	void getStartTime(void){
		getTime(startTime);
	}

	void getEndTime(void){

		getTime(endTime);
	}

	void getStartTimeIteration(void){
		getTime(startTimeIteration);
	}

	void getEndTimeIteration(void){

		getTime(endTimeIteration);
	}


	double elapsedSeconds = 0;
	double elapsedSecondsForSingleIteration = 0;

	void evaluateElapsedSecondsForOptimizationStudy(void){
		elapsedSeconds = std::difftime(endTime, startTime);

	}
	void evaluateElapsedSecondsForOptimizationIterationy(void){
		elapsedSecondsForSingleIteration = std::difftime(endTimeIteration, startTimeIteration);

	}


	std::string generateElapsedTime(double howManySeconds, string text) {
		int hours = static_cast<int>(howManySeconds) / 3600;
		int minutes = (static_cast<int>(howManySeconds) % 3600) / 60;
		double seconds = howManySeconds - (hours * 3600) - (minutes * 60);

		std::ostringstream oss;
		oss << text << ": " << hours << " hours, "
				<< minutes << " minutes, "
				<< seconds << " seconds";

		return oss.str();
	}


	std::string generateElapsedTimeStringForOptimizationIteration(void) {

		return generateElapsedTime(elapsedSecondsForSingleIteration, "Iteration run time");
	}


	std::string generateElapsedTimeStringForOptimization(void) {

		return generateElapsedTime(elapsedSeconds, "Optimization run time");
	}

};



class Optimizer {

private:

	OptimizationStatistics statistics;

	vec lowerBounds;
	vec upperBounds;

	vec lowerBoundsForAcqusitionFunctionMaximization;
	vec upperBoundsForAcqusitionFunctionMaximization;

	vec lowerBoundsForAcquisitionFunctionMaximizationGradientStep;
	vec upperBoundsForAcquisitionFunctionMaximizationGradientStep;

	double bestDeltaImprovementValueAchieved = 0.0;
	double bestFeasibleInitialValue = 0.0;


	bool ifAPIisUsed = false;
	bool ifTangentsAreUsed = false;
	bool WillGradientStepBePerformed = false;
	bool ifVariableSigmaStrategy = true;


	std::string designVectorFileName;

	const std::string currentDesignFileName = "currentDesign";

	OptimizationHistory history;

	std::vector<Design> lowFidelityDesigns;
	std::vector<Design> highFidelityDesigns;


	std::vector<ConstraintFunction> constraintFunctions;
	ObjectiveFunction objFun;

	std::vector<DesignForBayesianOptimization> theMostPromisingDesigns;


	unsigned int iterationNumberGradientStep = 0;
	unsigned int maximumIterationGradientStep = 5;
	double trustRegionFactorGradientStep = 1.0;
	double initialTrustRegionFactor = 1.0;


	unsigned int trSuccessCounter = 0.0;
	unsigned int trFailureCounter = 0.0;
	unsigned int trFailureTolerance = 15;
	unsigned int trSuccessTolerance = 3;
	double trLength = 0.8;
	double trLengthMin;
	double trLengthMax;


	GlobalOptimalDesign globalOptimalDesign;

	Design currentBestDesign;


	double factorForGradientStepWindow = 0.1;


	unsigned int iterMaxAcquisitionFunction;
	unsigned int iterMaxAcquisitionFunctionUnconstrainedGradientStep = 1000;
	unsigned int outerIterationNumber = 0;


	double sigmaFactor = 1.0;
	double sigmaFactorMax = 3.0;
	double sigmaFactorMin = 1.0;
	double crowdingCoefficient = 0.0;
	double sigmaMultiplier = 1.0;
	double sigmaGrowthFactor = 1.01;

	/* for integer design variables */
	unsigned int numberOfDiscreteVariables = 0;
	std::vector<double> incrementsForDiscreteVariables;
	std::vector<int> indicesForDiscreteVariables;


	unsigned int numberOfThreads = 1;

	unsigned int numberOfGlobalSearchSteps = 0;
	unsigned int numberOfLocalSearchSteps = 0;

	double improvementPercentThresholdForGradientStep;

	void initializeOptimizerSettings(void);

	void initializeOptimizationHistory(void);

	void setOptimizationHistoryConstraintsData(mat &historyData) const;

	void setOptimizationHistoryDataFeasibilityValues(mat &historyData) const;

	void calculateInitialImprovementValue(void);

	void findTheGlobalOptimalDesign(void);
	void initializeBoundsForAcquisitionFunctionMaximization();

	void modifySigmaFactor(void);



	void evaluateConstraints(Design &);

	bool areDiscreteParametersUsed(void) const;
	void roundDiscreteParameters(vec& designVector) const;
	vec generateDiscreteValues(unsigned int index, double dx) const;
	int findClosestDiscreteValue(double value, const vec& discreteValues) const;


	bool isConstrained(void) const;
	bool isNotConstrained(void) const;
	bool doesObjectiveFunctionHaveGradients(void) const;
	bool doesObjectiveFunctionHaveTangents(void) const;


	void modifyBoundsForInnerIterations(void);

	bool isToZoomInIteration(unsigned int) const;

	void trainSurrogatesForConstraints();


	void setOptimizationHistoryData(void);


	void estimateConstraints(DesignForBayesianOptimization &) const;
	void estimateConstraintsGradientStep(DesignForBayesianOptimization &design) const;

	//	double determineMaxStepSizeForGradientStep(vec x0, vec gradient) const;

	bool checkIfDesignIsWithinBounds(const vec &x0) const;

	void changeSettingsForAGradientBasedStep(void);
	void findTheMostPromisingDesignGradientStep(void);



	void findTheMostPromisingDesignToBeSimulated();
	void initializeCurrentBestDesign(void);
	void abortIfCurrentDesignHasANaN();
	void findPromisingDesignUnconstrainedGradientStep(
			DesignForBayesianOptimization &designToBeTried);
	bool checkIfBoxConstraintsAreSatisfied(const vec &dv) const;
	void decideIfAGradientStepShouldBeTakenForTheFirstIteration();
	void trimVectorSoThatItStaysWithinTheBounds(vec &x) const;
	void decideIfNextStepWouldBeAGradientStep();
	void adjustSigmaFactor(void);
	void printCurrentDesignToLogFile(void);
	void printGlobalOptimalDesignToLogFile(void);
	void searchAroundGlobalOptima(DesignForBayesianOptimization &designWithMaxAcqusition);
	void searchCompleteDesignSpace(DesignForBayesianOptimization &designWithMaxAcqusition);
	void scaleGradientVector(vec &gradient);
	void validateDimension() const;
	void validateBounds() const;
	void validateSizeIsEqualDimension(const vec &v) const;
	void findPromisingDesignConstrainedGradientStep(
			DesignForBayesianOptimization &designToBeTried);
	void validateFactorGradientStepWindow();

public:

	std::string name;

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maxNumberOfSamples = 0;

	unsigned int maxNumberOfSamplesLowFidelity = 0;

	unsigned int howOftenTrainModels = 100000;

	unsigned int iterGradientEILoop = 100;

	bool ifBoxConstraintsSet = false;
	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;


	Optimizer();
	Optimizer(const std::string&, unsigned int);


	void setDimension(unsigned int);
	void setName(std::string);
	void setImprovementPercentThresholdForGradientStep(double value);


	void setParameterToDiscrete(unsigned int, double);
	bool checkSettings(void) const;
	void print(void) const;
	void printSettingsToLogFile(void) const;
	void printIterationNumber(void) const;

	void printConstraints(void) const;

	void printHistory(void) const;
	mat getHistory(void) const;


	void performEfficientGlobalOptimization(void);
	void performEfficientGlobalOptimizationOnlyWithFunctionalValues(void);


	void initializeSurrogates(void);
	void trainSurrogates(void);

	void setMaximumNumberOfIterations(unsigned int );
	void setMaximumNumberOfIterationsLowFidelity(unsigned int);

	void setMaximumNumberOfInnerIterations(unsigned int);


	void setBoxConstraints(const Bounds& boxConstraints);

	void setFileNameDesignVector(std::string filename);

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setAPIUseOn(void);
	void setUseTangentsOn(void);
	void setParameterNames(std::vector<std::string> names);


	void setNumberOfThreads(unsigned int);


	void setHowOftenTrainModels(unsigned int value);

	void calculateImprovementValue(Design &);

	void addConstraint(ConstraintFunction &);


	void checkIfSettingsAreOK(void) const;
	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(const vec& constraintValues) const;

	void addObjectFunction(ObjectiveFunction &);


	void addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &) const;

	void computeConstraintsandPenaltyTerm(Design &);

	void addConstraintValuesToData(Design &d);

	void calculateFeasibilityProbabilities(DesignForBayesianOptimization &) const;

	double normalCdf(double value, double mu, double sigma) const;

	double calculateProbabilityLessThanAValue(double value, double mu, double sigma) const;
	double calculateProbabilityGreaterThanAValue(double value, double mu, double sigma) const;


	double calculateProbalityLessThanAValue(double value, double mu, double sigma) const;
	double calculateProbalityGreaterThanAValue(double value, double mu, double sigma) const;


	void defineTrustRegion(void);

	vec calculateGradientOfAcquisitionFunction(DesignForBayesianOptimization &) const;
	DesignForBayesianOptimization maximizeAcqusitionFunctionGradientBased(DesignForBayesianOptimization ) const;
	void findTheMostPromisingDesignEGO(void);
	DesignForBayesianOptimization getDesignWithMaxExpectedImprovement(void) const;

	void printInfoToLog(const string &msg) const{
		OptimizationLogger::getInstance().log(INFO,msg);
	}

	void printInfoToLog(const string &msg, double val) const{
		OptimizationLogger::getInstance().log(INFO,msg + std::to_string(val));
	}
	void printInfoToLog(const string &msg, unsigned int val) const{
		OptimizationLogger::getInstance().log(INFO,msg + std::to_string(val));
	}
	void printInfoToLog(const string &msg, int val) const{
		OptimizationLogger::getInstance().log(INFO,msg + std::to_string(val));
	}

	void printErrorToLog(const string &msg) const{
		OptimizationLogger::getInstance().log(ERROR,msg);
	}








};

} /* Namespace Rodop */

#endif
