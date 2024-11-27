#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include <limits> // For std::numeric_limits

#ifdef OPENMP_SUPPORT
#include <omp.h>
#endif


#include "./INCLUDE/optimization.hpp"
#include "./INCLUDE/aux.hpp"
#include "./INCLUDE/optimization_logger.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"

namespace Rodop{


Optimizer::Optimizer(){}


Optimizer::Optimizer(const std::string& nameTestcase, unsigned int numberOfOptimizationParams) {
	// Validate input parameters
	if (nameTestcase.empty()) {
		throw std::invalid_argument("Test case name cannot be empty.");
	}

	// Set the name and dimension
	name = nameTestcase;
	setDimension(numberOfOptimizationParams);
}


void Optimizer::initializeBoundsForAcquisitionFunctionMaximization() {

	if(dimension == 0){
		throw std::runtime_error("Dimension cannot be zero");
	}

	lowerBoundsForAcqusitionFunctionMaximization.resize(dimension);
	upperBoundsForAcqusitionFunctionMaximization.resize(dimension);
	upperBoundsForAcqusitionFunctionMaximization.fill(1.0 / dimension);

}

void Optimizer::initializeOptimizerSettings(void) {

	validateDimension();

	factorForGradientStepWindow = 0.01;
	/* maximum number of line search iterations */
	maximumIterationGradientStep = 5;

	sigmaFactor = 1.0;
	sigmaFactorMax = 100.0;
	sigmaFactorMin = 1.0;

	iterGradientEILoop = 100;
	improvementPercentThresholdForGradientStep = 5;

	trLengthMax = 1.0/dimension;
	trLengthMin = pow(0.5,8)*trLengthMax;
}






void Optimizer::setParameterToDiscrete(unsigned int index, double increment){

	assert(index <dimension);
	incrementsForDiscreteVariables.push_back(increment);
	indicesForDiscreteVariables.push_back(index);
	numberOfDiscreteVariables++;


}

bool Optimizer::checkSettings(void) const{

	printInfoToLog("Optimizer: Checking settings...");

	bool ifAllSettingsOk = true;

	if(!ifBoxConstraintsSet){

		ifAllSettingsOk = false;

	}

	return ifAllSettingsOk;
}






void Optimizer::addObjectFunction(ObjectiveFunction &objFunc){

	assert(ifObjectFunctionIsSpecied == false);

	objFun = objFunc;

	designVectorFileName = objFun.getFileNameDesignVector();

	objFun.setSigmaFactor(sigmaFactor);

	ifObjectFunctionIsSpecied = true;

}




bool Optimizer::checkBoxConstraints(void) const{

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) return false;
	}

	return true;
}


void Optimizer::print(void) const{

	std::cout<<"\nOptimizer Settings = \n\n";
	std::cout<<"Problem name : "<<name<<"\n";
	std::cout<<"Dimension    : "<<dimension<<"\n";
	std::cout<<"Number of threads    : "<<numberOfThreads<<"\n";
	std::cout<<"Maximum number of function evaluations: " <<maxNumberOfSamples<<"\n";
	std::cout<<"Maximum number of iterations for EI maximization: " << iterMaxAcquisitionFunction <<"\n";


	objFun.print();

	if (isNotConstrained()){
		std::cout << "Optimization problem does not have any constraints\n";
	}
	else{

		printConstraints();
	}

	//	if(numberOfDiscreteVariables > 0){
	//		std::cout << "Indices for discrete parameters = \n";
	//		printVector(indicesForDiscreteVariables);
	//		std::cout << "Incremental values for discrete parameters = \n";
	//		printVector(incrementsForDiscreteVariables);
	//	}

}


void Optimizer::printSettingsToLogFile(void) const{

	string msg = "\n";

	string tag = "Optimizer settings";

	string settingsMsg = generateFormattedString(tag,'=', 100) + "\n";
	settingsMsg+= "Problem name : " + name + "\n";
	settingsMsg+= "Dimension    : " + std::to_string(dimension) + "\n";
	settingsMsg+= "Number of threads    : " + std::to_string(numberOfThreads) + "\n";
	settingsMsg+= "Maximum number of function evaluations    : " + std::to_string(maxNumberOfSamples) + "\n";

	std::string border(100, '=');
	settingsMsg += border + "\n";

	msg += settingsMsg;


	string objectiveFunctionMsg = objFun.generateOutputString();

	msg += objectiveFunctionMsg;

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		string constraintFunctionMsg = it->generateOutputString();
		msg += constraintFunctionMsg;
	}

	msg +="\n";

	printInfoToLog(msg);

}


void Optimizer::initializeSurrogates() {

	// Ensure the objective function is specified before proceeding
	if (!ifObjectFunctionIsSpecied) {
		throw std::runtime_error("Error: Objective function is not specified. Cannot initialize surrogates.");
	}

	printInfoToLog("Initializing surrogate model for the objective function...");
	objFun.initializeSurrogate();

	// Loop through each constraint function to initialize its surrogate model if needed
	for (auto& constraint : constraintFunctions) {
		printInfoToLog("Initializing surrogate model for a constraint...");

		// Only initialize surrogate if the constraint is not user-defined
		if (!constraint.isUserDefinedFunction()) {
			constraint.initializeSurrogate();
		} else {
			printInfoToLog("Skipping user-defined function for surrogate initialization.");
		}
	}

	// Mark surrogates as initialized
	ifSurrogatesAreInitialized = true;

	printInfoToLog("Surrogate model initialization completed.");
}



void Optimizer::trainSurrogates(void){
	printInfoToLog("Training surrogate model for the objective function...");
	objFun.trainSurrogate();

	if(isConstrained()){
		trainSurrogatesForConstraints();
	}
}


void Optimizer::addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &designCalculated) const{

	if(isConstrained()){

		estimateConstraints(designCalculated);

		calculateFeasibilityProbabilities(designCalculated);

		designCalculated.updateAcqusitionFunctionAccordingToConstraints();

	}
}

bool Optimizer::doesObjectiveFunctionHaveGradients(void) const{

	assert(ifObjectFunctionIsSpecied);
	SURROGATE_MODEL type = objFun.getSurrogateModelType();

	if (type == GRADIENT_ENHANCED) {

		return true;
	}
	else{
		return false;
	}

}


bool Optimizer::doesObjectiveFunctionHaveTangents(void) const{

	assert(ifObjectFunctionIsSpecied);
	SURROGATE_MODEL type = objFun.getSurrogateModelType();

	if (type == TANGENT_ENHANCED) {

		return true;
	}
	else{
		return false;
	}

}




bool Optimizer::areDiscreteParametersUsed(void) const{

	if(numberOfDiscreteVariables>0) return true;
	else return false;
}



void Optimizer::checkIfSettingsAreOK() const {

	// Check if the maximum number of samples is set
	if (maxNumberOfSamples == 0) {
		throw std::runtime_error("Error: Maximum number of samples is not set for the optimization.");
	}

	// Check if box constraints are set correctly
	if (!checkBoxConstraints()) {
		throw std::runtime_error("Error: Box constraints are not set properly.");
	}
}


void Optimizer::findTheGlobalOptimalDesign() {

	// Ensure that box constraints are set
	if (!ifBoxConstraintsSet) {
		throw std::runtime_error("Error: Box constraints are not set.");
	}

	// Retrieve history data
	mat historyData = history.getData();
	if (historyData.getNRows() == 0 || historyData.getNCols() == 0) {
		throw std::runtime_error("Error: History data is empty. Cannot find global optimal design.");
	}

	printInfoToLog("Finding global optimal design from history data.");

	// Set the global optimal design from the history file
	globalOptimalDesign.setGlobalOptimalDesignFromHistoryFile(historyData);

	// Save the global optimal design to an XML file
	try {
		globalOptimalDesign.saveToXMLFile();
		printInfoToLog("Global optimal design saved to XML file.");
	} catch (const std::exception& e) {
		OptimizationLogger::getInstance().log(ERROR, std::string("Error saving global optimal design to XML: ") + e.what());
		throw;
	}

	printInfoToLog("Global optimal design successfully found and saved.");
}

void Optimizer::validateFactorGradientStepWindow() {
	// Validate input values
	if (factorForGradientStepWindow <= 0.0
			|| factorForGradientStepWindow > 1.0) {
		throw std::invalid_argument(
				"Factor for gradient step window must be between 0 and 1.");
	}
}


void Optimizer::defineTrustRegion(void) {

	validateDimension();
	validateSizeIsEqualDimension(globalOptimalDesign.designParametersNormalized);

	vec dv = globalOptimalDesign.designParametersNormalized;

	if(dv.getSize() == 0){
		throw std::runtime_error("Global optimal design is undefined");
	}

	double delta = (1.0 / dimension) * trLength;

	// Set lower and upper bounds
	lowerBoundsForAcqusitionFunctionMaximization = dv - delta;
	upperBoundsForAcqusitionFunctionMaximization = dv + delta;

	trimVectorSoThatItStaysWithinTheBounds(lowerBoundsForAcqusitionFunctionMaximization);
	trimVectorSoThatItStaysWithinTheBounds(upperBoundsForAcqusitionFunctionMaximization);

	printInfoToLog("Lower bounds for inner optimization = " + lowerBoundsForAcqusitionFunctionMaximization.toString());
	printInfoToLog("Upper bounds for inner optimization = " + upperBoundsForAcqusitionFunctionMaximization.toString());

}

void Optimizer::changeSettingsForAGradientBasedStep(void) {

	validateFactorGradientStepWindow();
	validateDimension();  // Ensure dimension is properly initialized
	validateSizeIsEqualDimension(globalOptimalDesign.designParametersNormalized);

	vec dv = globalOptimalDesign.designParametersNormalized;

	// Calculate delta for gradient-based step
	double delta = (factorForGradientStepWindow / dimension) * trustRegionFactorGradientStep;

	// Set lower and upper bounds
	lowerBoundsForAcquisitionFunctionMaximizationGradientStep = dv - delta;
	upperBoundsForAcquisitionFunctionMaximizationGradientStep = dv + delta;

	trimVectorSoThatItStaysWithinTheBounds(lowerBoundsForAcquisitionFunctionMaximizationGradientStep);
	trimVectorSoThatItStaysWithinTheBounds(upperBoundsForAcquisitionFunctionMaximizationGradientStep);
}


//double Optimizer::determineMaxStepSizeForGradientStep(vec x0, vec gradient) const {
//
//	// Validate sizes
//	validateDimension();
//	validateSizeIsEqualDimension(upperBoundsForAcquisitionFunctionMaximizationGradientStep);
//	validateSizeIsEqualDimension(lowerBoundsForAcquisitionFunctionMaximizationGradientStep);
//	validateSizeIsEqualDimension(x0);
//	validateSizeIsEqualDimension(gradient);
//
//	vec ub = upperBoundsForAcquisitionFunctionMaximizationGradientStep;
//	vec lb = lowerBoundsForAcquisitionFunctionMaximizationGradientStep;
//
//	vec stepSizeTemp(dimension);
//	double stepSizeMax = 0.0;
//
//	printInfoToLog("Determining the max step size for the gradient step");
//	printInfoToLog("Gradient = " + gradient.toString());
//	printInfoToLog("Lower bounds = " + lb.toString());
//	printInfoToLog("Upper bounds = " + ub.toString());
//
//
//	// Loop over each dimension to determine the maximum feasible step size
//	for (unsigned int i = 0; i < dimension; i++) {
//
//		double step = 0.0;
//
//		// Handle zero gradient to avoid division by zero
//		if (fabs(gradient(i)) < 1e-12) {
//			stepSizeTemp(i) = std::numeric_limits<double>::infinity(); // No movement in this direction
//
//			printInfoToLog(std::to_string(i) + "th gradient entry is zero or very small: " + std::to_string(gradient(i)));
//			continue;
//		}
//
//		if (gradient(i) > 0.0) {  // Gradient positive, move towards lower bound
//			step = -(lb(i) - x0(i)) / gradient(i);
//		} else {  // Gradient negative move towards upper bound
//			step = -(ub(i) - x0(i)) / gradient(i);
//		}
//
//
//		stepSizeTemp(i) = step;
//	}
//
//	printInfoToLog("Step sizes = " + stepSizeTemp.toString(15));
//	// Find the smallest positive step size that respects the bounds
//	stepSizeMax = stepSizeTemp.findMin();
//
//	return stepSizeMax;
//}


void Optimizer::trimVectorSoThatItStaysWithinTheBounds(vec &x) const{

	validateDimension();
	double oneOverDim = 1.0 / dimension;
	for (unsigned int i = 0; i < dimension; i++) {
		if (x(i) < 0.0) {
			x(i) = 0.0;
		}
		if (x(i) > oneOverDim) {
			x(i) = oneOverDim;
		}
	}
}

void Optimizer::scaleGradientVector(vec &gradient) {
	for (unsigned int i = 0; i < dimension; i++) {
		double scalingFactor = (upperBounds(i) - lowerBounds(i)) * dimension;
		if (scalingFactor <= 0.0)
			throw std::runtime_error("Scaling factor must be positive.");

		gradient(i) *= scalingFactor;
	}
}

void Optimizer::validateDimension() const{
	// Validate dimensions and bounds
	if (dimension == 0)
		throw std::runtime_error("Invalid dimension.");
}

void Optimizer::validateBounds() const{
	if (lowerBounds.getSize() != dimension
			|| upperBounds.getSize() != dimension) {
		throw std::runtime_error(
				"Dimension of bounds does not match the design space.");
	}
}

void Optimizer::validateSizeIsEqualDimension(const vec &gradient) const{
	if (gradient.getSize() != dimension) {
		throw std::runtime_error("Vector size does not match dimension.");
	}
}

void Optimizer::decideIfNextStepWouldBeAGradientStep() {
	if (iterationNumberGradientStep == maximumIterationGradientStep) {
		printInfoToLog("Maximum number of gradient-based iterations...");
		trustRegionFactorGradientStep = 1.0;
		initialTrustRegionFactor = 1.0;
		printInfoToLog("Trust region factor = " + std::to_string(trustRegionFactorGradientStep));
		iterationNumberGradientStep = 0;
		WillGradientStepBePerformed = false;

	}
}


void Optimizer::findPromisingDesignUnconstrainedGradientStep(DesignForBayesianOptimization &designToBeTried) {

	vec gradient = globalOptimalDesign.gradient;
	validateSizeIsEqualDimension(gradient);
	scaleGradientVector(gradient);

	vec lb = lowerBoundsForAcquisitionFunctionMaximizationGradientStep;
	vec ub = upperBoundsForAcquisitionFunctionMaximizationGradientStep;
	validateSizeIsEqualDimension(lb);
	validateSizeIsEqualDimension(ub);

	printInfoToLog("Trust region factor = ", trustRegionFactorGradientStep);
	printInfoToLog("Gradient search window (unconstrained)");
	printInfoToLog("lb = " + lb.toString());
	printInfoToLog("ub = " + ub.toString());

	double maxEI = -LARGE;
	DesignForBayesianOptimization designWithMaxImprovement(dimension);
	int numberOfInnerIterations = static_cast<int>(iterMaxAcquisitionFunction);
	printInfoToLog("Number of inner iterations = ",  numberOfInnerIterations);
	printInfoToLog("Number of threads = ",  numberOfThreads);


#ifdef OPENMP_SUPPORT
	omp_set_num_threads(numberOfThreads);
#pragma omp parallel for
#endif

	for (int i = 0; i < numberOfInnerIterations; i++) {
		/* generate a design around the global optimal design */
		designToBeTried.generateRandomDesignVector(lb, ub);
		vec d = designToBeTried.dv - globalOptimalDesign.designParametersNormalized;
		//				d.print("d");
		/* x_k+1 = x_k + d  =>  d = x_k+1 - x_k */
		double gradDotDv = d.dot(gradient);
		if (gradDotDv < 0) {
			//					designToBeTried.dv.print("dv random");
			objFun.calculateExpectedImprovementUsingDerivatives(designToBeTried);
			if (designToBeTried.valueAcquisitionFunction > maxEI) {
				{
					maxEI = designToBeTried.valueAcquisitionFunction;
					designWithMaxImprovement = designToBeTried;
					//							printScalar(maxImprovement);
					//					p.print("feasibility probabilities");
				}
			}
		}
	} /*end of for loop */
	//			printScalar(maxImprovement);
	designToBeTried = designWithMaxImprovement;
	printInfoToLog(designToBeTried.toString());
}


void Optimizer::findPromisingDesignConstrainedGradientStep(DesignForBayesianOptimization &designToBeTried) {

	vec gradient = globalOptimalDesign.gradient;
	validateSizeIsEqualDimension(gradient);
	scaleGradientVector(gradient);

	vec lb = lowerBoundsForAcquisitionFunctionMaximizationGradientStep;
	vec ub = upperBoundsForAcquisitionFunctionMaximizationGradientStep;
	validateSizeIsEqualDimension(lb);
	validateSizeIsEqualDimension(ub);

	printInfoToLog("Trust region factor = ", trustRegionFactorGradientStep);
	printInfoToLog("Gradient search window (constrained)");
	printInfoToLog("lb = " + lb.toString());
	printInfoToLog("ub = " + ub.toString());

	double maxImprovement = -LARGE;
	DesignForBayesianOptimization designWithMaxImprovement(dimension,numberOfConstraints);
	int numberOfInnerIterations = static_cast<int>(iterMaxAcquisitionFunction);
	printInfoToLog("Number of inner iterations = ",  numberOfInnerIterations);
	printInfoToLog("Number of threads = ",  numberOfThreads);


#ifdef OPENMP_SUPPORT
	omp_set_num_threads(numberOfThreads);
#pragma omp parallel for
#endif
	for (int i = 0; i < numberOfInnerIterations; i++) {
		/* generate a design around the global optimal design */
		designToBeTried.generateRandomDesignVector(lb, ub);
		vec d = designToBeTried.dv - globalOptimalDesign.designParametersNormalized;
		//				d.print("d");
		/* x_k+1 = x_k + d  =>  d = x_k+1 - x_k */
		double gradDotDv = d.dot(gradient);
		if (gradDotDv < 0) {
			//					designToBeTried.dv.print("dv random");
			objFun.calculateExpectedImprovementUsingDerivatives(designToBeTried);
			/* improvement is either zero or a positive value */
			double ExpectedImprovement = designToBeTried.valueAcquisitionFunction;

			estimateConstraints(designToBeTried);
			calculateFeasibilityProbabilities(designToBeTried);
			vec p = designToBeTried.constraintFeasibilityProbabilities;
			for (unsigned int j = 0; j < numberOfConstraints; j++) {
				ExpectedImprovement *= p(j);
			}
			//				designToBeTried.print();
			//				printScalar(improvement);
			if (ExpectedImprovement > maxImprovement) {
				{
					maxImprovement = ExpectedImprovement;
					designWithMaxImprovement = designToBeTried;
					//							printScalar(maxImprovement);
					//					p.print("feasibility probabilities");
				}
			}
		}
	} /*end of for loop */
	//			printScalar(maxImprovement);
	designToBeTried = designWithMaxImprovement;
	printInfoToLog(designToBeTried.toString());
}

void Optimizer::findTheMostPromisingDesignGradientStep(void){

	printInfoToLog("Finding the most promising design using a gradient based local search...");

	globalOptimalDesign.setGradientGlobalOptimumFromTrainingData(objFun.getFileNameTrainingData());

	printInfoToLog(globalOptimalDesign.toString());

	bool ifGradientVectorExists = globalOptimalDesign.checkIfGlobalOptimaHasGradientVector();

	if(!ifGradientVectorExists){
		printInfoToLog("Global optimal design has no gradient.");
		WillGradientStepBePerformed = false;
	}

	if(WillGradientStepBePerformed){

		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);

		/* This function adjust the bounds for the gradient based search, in other words, it specifies the size of the
		 * search window in the gradient based step.
		 */
		changeSettingsForAGradientBasedStep();

		if(isConstrained()){
			findPromisingDesignConstrainedGradientStep(designToBeTried);
		} /* end of ifConstrained */

		else{
			findPromisingDesignUnconstrainedGradientStep(designToBeTried);
		}

		/* we reduce the local search window size and line search iteration number */
		theMostPromisingDesigns.push_back(designToBeTried);
		iterationNumberGradientStep++;
		trustRegionFactorGradientStep = trustRegionFactorGradientStep * 0.5;

		decideIfNextStepWouldBeAGradientStep();

		numberOfLocalSearchSteps++;
	}
	else{
		printInfoToLog("No gradient-based step, back to default EGO step");
		/* fallback solution if a gradient-based step cannot be done */
		findTheMostPromisingDesignEGO();
	}

}

void Optimizer::searchAroundGlobalOptima(DesignForBayesianOptimization &designWithMaxAcqusition) {
	/* Search around the global design if possible */

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;

	vec dvNotNormalized =globalOptimalDesign.designParametersNormalized.denormalizeVector(lowerBounds, upperBounds);
	if (checkIfBoxConstraintsAreSatisfied(dvNotNormalized)) {
		printInfoToLog("Local search around global design...");
		printInfoToLog("Global optimal design = " + dvNotNormalized.toString());

		int numberOfInnerIterations = static_cast<int>(iterMaxAcquisitionFunction);

#ifdef OPENMP_SUPPORT
#pragma omp parallel for
#endif

		for (int i = 0; i < numberOfInnerIterations; i++) {
			vec dv0 = globalOptimalDesign.designParametersNormalized;
			DesignForBayesianOptimization designToBeTried(dimension,
					numberOfConstraints);
			designToBeTried.generateRandomDesignVectorAroundASample(dv0, lb,ub);
			objFun.calculateExpectedImprovement(designToBeTried);
			addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);
			if (designToBeTried.valueAcquisitionFunction
					> designWithMaxAcqusition.valueAcquisitionFunction) {
#ifdef OPENMP_SUPPORT
#pragma omp critical
#endif
				{
					designWithMaxAcqusition = designToBeTried;
				}
#if 0
				printf("A design with a better EI value has been found (second loop) \n");
				designToBeTried.print();
#endif
			}
		}
	}
}

void Optimizer::searchCompleteDesignSpace(DesignForBayesianOptimization &designWithMaxAcqusition) {

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;
	int numberOfInnerIterations = static_cast<int>(iterMaxAcquisitionFunction);

	if(lb.getSize() != ub.getSize() || lb.getSize() != dimension){

		throw std::runtime_error("Optimizer::findTheMostPromisingDesignEGO: There is some error with lower and upper bounds.");
	}
	printInfoToLog("Global search in design space..");


#ifdef OPENMP_SUPPORT
#pragma omp parallel for
#endif

	for (int i = 0; i < numberOfInnerIterations; i++) {
		DesignForBayesianOptimization designToBeTried(dimension,
				numberOfConstraints);
		designToBeTried.generateRandomDesignVector(lb, ub);
		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);
		if (designToBeTried.valueAcquisitionFunction
				> designWithMaxAcqusition.valueAcquisitionFunction) {
#ifdef OPENMP_SUPPORT
#pragma omp critical
#endif
			{
				designWithMaxAcqusition = designToBeTried;
			}
		}
	}
}

/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void Optimizer::findTheMostPromisingDesignEGO(void){

	if (!ifSurrogatesAreInitialized || globalOptimalDesign.designParametersNormalized.getSize() == 0) {
		throw std::runtime_error("Surrogates are not initialized or the global optimal design is invalid.");
	}
	validateDimension();

	defineTrustRegion();

	printInfoToLog("Searching the best potential design (inner optimization loop)...");
	double bestFeasibleObjectiveFunctionValue = globalOptimalDesign.trueValue;


	if(globalOptimalDesign.isDesignFeasible == false){

		bestFeasibleObjectiveFunctionValue = LARGE;
	}
	printInfoToLog("Best feasible objective value = " + std::to_string(bestFeasibleObjectiveFunctionValue));

	objFun.setFeasibleMinimum(bestFeasibleObjectiveFunctionValue);

	DesignForBayesianOptimization designWithMaxAcqusition(dimension,numberOfConstraints);
	designWithMaxAcqusition.valueAcquisitionFunction = -LARGE;
	printInfoToLog("Number of inner iterations = " + std::to_string(iterMaxAcquisitionFunction));

#ifdef OPENMP_SUPPORT
	omp_set_num_threads(numberOfThreads);
	printInfoToLog("Number of threads used = " + std::to_string(numberOfThreads));
#endif


	searchCompleteDesignSpace(designWithMaxAcqusition);
	searchAroundGlobalOptima(designWithMaxAcqusition);

	printInfoToLog(designWithMaxAcqusition.toString());

	theMostPromisingDesigns.push_back(designWithMaxAcqusition);
	numberOfGlobalSearchSteps++;

}


DesignForBayesianOptimization Optimizer::getDesignWithMaxExpectedImprovement(void) const{
	return theMostPromisingDesigns.front();
}


vec Optimizer::calculateGradientOfAcquisitionFunction(DesignForBayesianOptimization &currentDesign) const {

	if (dimension == 0) {
		throw std::runtime_error("Dimension cannot be zero.");
	}
	if (!ifSurrogatesAreInitialized) {
		throw std::runtime_error("Surrogate models must be initialized first.");
	}

	//	printInfoToLog("Calculating the gradient of the acquisition function...");

	vec gradient(dimension);

	for (unsigned int i = 0; i < dimension; i++) {

		double dvSave = currentDesign.dv(i);

		// Define epsilon as a small percentage of the design variable or a small absolute minimum to avoid zero
		double epsilon = std::max(currentDesign.dv(i) * 0.00001, 1e-8);

		// Central finite difference approximation
		currentDesign.dv(i) += epsilon;
		objFun.calculateExpectedImprovement(currentDesign);
		double EIplus = currentDesign.valueAcquisitionFunction;

		currentDesign.dv(i) -= 2 * epsilon;
		objFun.calculateExpectedImprovement(currentDesign);
		double EIminus = currentDesign.valueAcquisitionFunction;

		// Central difference quotient for the gradient
		double fdVal = (EIplus - EIminus) / (2 * epsilon);
		gradient(i) = fdVal;

		// Restore the original value of the design variable
		currentDesign.dv(i) = dvSave;
	}

	return gradient;
}



DesignForBayesianOptimization Optimizer::maximizeAcqusitionFunctionGradientBased(DesignForBayesianOptimization initialDesign) const {


	printInfoToLog("Maximizing acquisition function using gradient based search...");
	vec grad(dimension);
	double stepSize0 = 0.001;
	double stepSize = 0.0;


	objFun.calculateExpectedImprovement(initialDesign);
	addPenaltyToAcqusitionFunctionForConstraints(initialDesign);


	double f0 = initialDesign.valueAcquisitionFunction;
	DesignForBayesianOptimization bestDesign = initialDesign;

	bool breakOptimization = false;

	for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){

		//		printInfoToLog("Gradient search iteration = " + std::to_string(iterGradientSearch));

		grad = calculateGradientOfAcquisitionFunction(bestDesign);

		/* save the design vector */
		DesignForBayesianOptimization dvLineSearchSave = bestDesign ;

		//		printInfoToLog("Starting line search...");
		stepSize = stepSize0;

		while(1){ /* Here we start the line search */


			/* design update */

			vec lb = lowerBoundsForAcqusitionFunctionMaximization;
			vec ub = upperBoundsForAcqusitionFunctionMaximization;

			bestDesign.gradientUpdateDesignVector(grad,lb,ub,stepSize);

			objFun.calculateExpectedImprovement(bestDesign);
			addPenaltyToAcqusitionFunctionForConstraints(bestDesign);

			//			printInfoToLog("EI value = " + std::to_string(bestDesign.valueAcquisitionFunction));

			/* if ascent is achieved */
			if(bestDesign.valueAcquisitionFunction > f0){
				//				printInfoToLog("Ascent is achieved.");
				//				printInfoToLog(bestDesign.toString());

				f0 = bestDesign.valueAcquisitionFunction;
				break;
			}

			else{ /* else halve the stepsize and set design to initial */

				stepSize = stepSize * 0.5;
				bestDesign = dvLineSearchSave;

				if(stepSize < 10E-12) {
					//					printInfoToLog("The stepsize is getting too small. Optimization will terminate.");

					breakOptimization = true;
					break;
				}
			}

		} /* Line search loop */

		if(breakOptimization) break;

	} /* end of gradient-search loop */

	printInfoToLog(bestDesign.toString());
	return bestDesign;

}


void Optimizer::setOptimizationHistoryConstraintsData(mat& historyData) const {

	// Ensure historyData is valid and dimension is set
	if (historyData.isEmpty()) {
		OptimizationLogger::getInstance().log(ERROR, "Error: historyData is empty.");
		throw std::invalid_argument("Error: historyData is empty.");
	}

	if (dimension <= 0) {
		OptimizationLogger::getInstance().log(ERROR, "Error: dimension must be greater than 0.");
		throw std::invalid_argument("Error: dimension must be greater than 0.");
	}

	unsigned int N = historyData.getNRows();
	printInfoToLog("Setting optimization history constraints data for " + std::to_string(N) + " rows.");

	mat inputObjectiveFunction = historyData.submat(0, N - 1, 0, dimension - 1);

	for (const auto& constraint : constraintFunctions) {
		int ID = constraint.getID();
		if (ID < 0 || ID >= static_cast<int>(numberOfConstraints)) {
			OptimizationLogger::getInstance().log(ERROR, "Error: Constraint ID " + std::to_string(ID) + " is out of range.");
			throw std::out_of_range("Error: Constraint ID is out of range.");
		}

		printInfoToLog("Processing constraint ID: " + std::to_string(ID));

		if (!constraint.isUserDefinedFunction()) {
			// Non-user-defined constraint, read training data
			mat dataRead = constraint.getTrainingData();
			mat inputConstraint = dataRead.submat(0, dataRead.getNRows() - 1, 0, dimension - 1);
			std::string type = constraint.getInequalityType();
			printInfoToLog("Non-user-defined constraint, inequality type: " + type);

			// Update historyData based on training data
			for (unsigned int i = 0; i < N; ++i) {
				vec input = inputObjectiveFunction.getRow(i);
				int indx = inputConstraint.findRowIndex(input, 1e-8);

				if (indx >= 0) {
					historyData(i, dimension + ID + 1) = dataRead(indx, dimension);
					printInfoToLog("Row is found in the objective function training data. \n x = " + input.toString() + "\n");
				} else {
					printInfoToLog("Row is not found in the objective function training data. \n x = " + input.toString() + "\n");
					if (type == ">") {
						historyData(i, dimension + ID + 1) = -1e15;
					} else if (type == "<") {
						historyData(i, dimension + ID + 1) = 1e15;
					}
				}
			}

		} else {
			// User-defined constraint
			printInfoToLog("User-defined constraint found.");

			for (unsigned int i = 0; i < N; ++i) {
				vec x = inputObjectiveFunction.getRow(i);
				double constraintValue = constraint.callUserDefinedFunction(x);
				historyData(i, dimension + ID + 1) = constraintValue;
				printInfoToLog("User-defined constraint evaluated for row " + std::to_string(i) + ", value: " + std::to_string(constraintValue));
			}
		}
	}

	printInfoToLog("Finished setting optimization history constraints data.");
}




bool Optimizer::checkIfBoxConstraintsAreSatisfied(const vec &designVariables) const {

	// Ensure that the box constraints are set and the input size is valid
	if (!ifBoxConstraintsSet) {
		throw std::runtime_error("Error: Box constraints are not set.");
	}

	if (designVariables.getSize() == 0 || designVariables.getSize() != dimension || designVariables.getSize() != lowerBounds.getSize()) {
		throw std::invalid_argument("Error: Design variables size does not match dimension or bounds size.");
	}

	// Check if each design variable satisfies the box constraints
	for (unsigned int i = 0; i < dimension; ++i) {
		if (designVariables(i) < lowerBounds(i) || designVariables(i) > upperBounds(i)) {
			// Log the violation
			std::ostringstream oss;
			oss << "Box constraint violation for variable " << i << ": "
					<< designVariables(i) << " not in [" << lowerBounds(i) << ", " << upperBounds(i) << "]";
			OptimizationLogger::getInstance().log(WARNING, oss.str());

			return false;  // Return false as soon as a violation is found
		}
	}

	// If no violations are found, return true
	return true;
}

void Optimizer::setOptimizationHistoryDataFeasibilityValues(mat &historyData) const {

	// Check that the historyData has valid rows and dimension is set
	if (historyData.getNRows() == 0) {
		throw std::invalid_argument("HistoryData has no rows.");
	}

	if (dimension == 0) {
		throw std::invalid_argument("Dimension must be greater than 0.");
	}

	unsigned int N = historyData.getNRows();
	unsigned int nCols = historyData.getNCols();

	if(isConstrained()) {

		printInfoToLog("Evaluating feasibility for " + std::to_string(N) + " samples.");

		// Loop through each row in the historyData
		for (unsigned int i = 0; i < N; ++i) {

			printInfoToLog("Evaluating the feasibility of the sample " + std::to_string(i) + ": Feasible.");

			vec rowOfTheHistoryFile = historyData.getRow(i);
			vec dv = rowOfTheHistoryFile.head(dimension);

			// Assume the design is feasible initially
			bool isFeasible = true;

			// Check constraint feasibility
			if (isConstrained()) {
				vec constraintValues(numberOfConstraints);
				for (unsigned int j = 0; j < numberOfConstraints; ++j) {
					constraintValues(j) = rowOfTheHistoryFile(j + dimension + 1);
				}
				isFeasible = checkConstraintFeasibility(constraintValues);
			}

			// Check if the design vector satisfies box constraints
			bool ifBoxConstraintsAreSatisfied = checkIfBoxConstraintsAreSatisfied(dv);

			// Set the feasibility value in the last column of historyData
			if (isFeasible && ifBoxConstraintsAreSatisfied) {
				historyData(i, nCols - 1) = 1.0;  // Feasible
				printInfoToLog("Sample " + std::to_string(i) + ": Feasible.");
			} else {
				historyData(i, nCols - 1) = 0.0;  // Not feasible
				printInfoToLog("Sample " + std::to_string(i) + ": Not Feasible.");
			}
		}

		printInfoToLog("Completed feasibility evaluation.");

	}
	else{
		for (unsigned int i = 0; i < N; ++i) {
			historyData(i, nCols - 1) = 1.0;
		}
	}




}



void Optimizer::initializeOptimizationHistory() {

	// Ensure that the optimizer is properly configured
	if (dimension == 0) {
		throw std::runtime_error("Error: Dimension must be greater than 0.");
	}

	if (!ifObjectFunctionIsSpecied) {
		throw std::runtime_error("Error: Objective function is not specified.");
	}

	if (!ifSurrogatesAreInitialized) {
		throw std::runtime_error("Error: Surrogates are not initialized.");
	}

	// Set the dimension for the history
	history.setDimension(dimension);

	// Set the name of the objective function in the history
	history.setObjectiveFunctionName(objFun.getName());

	// If constrained, add constraint names to the history
	if (isConstrained()) {
		for (const auto& constraint : constraintFunctions) {
			history.addConstraintName(constraint.getName());
		}
	}

	// Set the optimization history data
	setOptimizationHistoryData();

	// Calculate the crowding factor if the variable sigma strategy is enabled
	if (ifVariableSigmaStrategy) {
		history.calculateCrowdingFactor();
		crowdingCoefficient = history.getCrowdingFactor() * sigmaFactor;
	}
}

void Optimizer::setOptimizationHistoryData(void) {

	if (!ifSurrogatesAreInitialized) {
		throw std::runtime_error("Surrogate models are not initialized.");
	}
	if (!ifObjectFunctionIsSpecied) {
		throw std::runtime_error("Objective function is not specified.");
	}
	if (dimension == 0) {
		throw std::runtime_error("Dimension must be greater than zero.");
	}

	printInfoToLog("setting Optimization history data...");

	// Retrieve the training data from the objective function
	mat trainingDataObjectiveFunction = objFun.getTrainingData();
	unsigned int N = trainingDataObjectiveFunction.getNRows();

	// Ensure that trainingDataObjectiveFunction has enough columns
	if (dimension > trainingDataObjectiveFunction.getNCols() - 1) {
		throw std::out_of_range("Dimension exceeds the number of columns in the training data.");
	}

	// Submatrix for the input part of the objective function training data
	mat inputObjectiveFunction = trainingDataObjectiveFunction.submat(0, N-1, 0, dimension-1);

	unsigned int numberOfEntries = dimension + 1 + numberOfConstraints + 2;
	mat optimizationHistoryData(N, numberOfEntries);

	// Copy the relevant columns from the training data
	for (unsigned int i = 0; i < dimension + 1; i++) {
		vec temp = trainingDataObjectiveFunction.getCol(i);
		optimizationHistoryData.setCol(temp, i);
	}

	// Handle constraints if they exist
	if (isConstrained()) {
		setOptimizationHistoryConstraintsData(optimizationHistoryData);
	}

	// Set feasibility values
	setOptimizationHistoryDataFeasibilityValues(optimizationHistoryData);

	// Set data into the history object

	//   optimizationHistoryData.print();

	history.setData(optimizationHistoryData);

	// Calculate the best feasible initial value and log it
	bestFeasibleInitialValue = history.calculateInitialImprovementValue();
	printInfoToLog("Best feasible initial value: " + std::to_string(bestFeasibleInitialValue));

	// Save optimization history data to a file
	history.saveOptimizationHistoryFile();
	history.numberOfDoESamples = N;

	// Find the global optimal design based on history data
	findTheGlobalOptimalDesign();
}


void Optimizer::findTheMostPromisingDesignToBeSimulated() {

	DesignForBayesianOptimization optimizedDesignGradientBased;

	theMostPromisingDesigns.clear();

	if (doesObjectiveFunctionHaveGradients() && WillGradientStepBePerformed) {

		printInfoToLog("A Gradient Step will be performed...");
		printInfoToLog("Gradient search iteration number = " + std::to_string(iterationNumberGradientStep));
		findTheMostPromisingDesignGradientStep();

		optimizedDesignGradientBased = theMostPromisingDesigns.at(0);


	} else {
		printInfoToLog("An EGO Step will be performed...");
		findTheMostPromisingDesignEGO();
		optimizedDesignGradientBased = maximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));

	}

	vec best_dvNorm = optimizedDesignGradientBased.dv;
	vec best_dv = best_dvNorm.denormalizeVector(lowerBounds, upperBounds);

	if(areDiscreteParametersUsed()){
		roundDiscreteParameters(best_dv);
	}

	currentBestDesign.designParametersNormalized = best_dvNorm;
	currentBestDesign.designParameters = best_dv;

}

void Optimizer::initializeCurrentBestDesign(void) {
	currentBestDesign.tag = "Current Iterate";
	currentBestDesign.setNumberOfConstraints(numberOfConstraints);

	if(!ifAPIisUsed){
		currentBestDesign.saveDesignVector(designVectorFileName);
	}
	currentBestDesign.isDesignFeasible = true;
}

void Optimizer::abortIfCurrentDesignHasANaN() {
	if (currentBestDesign.checkIfHasNan()) {
		throw std::runtime_error("Error: NaN detected in current best design while reading external executable outputs.");
	}
}


void Optimizer::decideIfAGradientStepShouldBeTakenForTheFirstIteration() {

	findTheGlobalOptimalDesign();

	if (doesObjectiveFunctionHaveGradients()) {
		globalOptimalDesign.setGradientGlobalOptimumFromTrainingData(objFun.getFileNameTrainingData());

		if (globalOptimalDesign.checkIfGlobalOptimaHasGradientVector()) {

			if(globalOptimalDesign.isDesignFeasible){
				printInfoToLog("Global optimal design is feasible and has gradient, the next step will be a gradient search step.");
				WillGradientStepBePerformed = true;
			}
			else{
				printInfoToLog("Global optimal design has gradient but is not feasible and has gradient, the next step will be a global search step.");
			}

		}
		else{
			printInfoToLog("Global optimal design has no gradient, the next step will be a global search step.");
		}
	}
}



void Optimizer::adjustSigmaFactor(void) {
	history.calculateCrowdingFactor();
	double cFactor = history.getCrowdingFactor();
	//	output.printMessage("cFactor", cFactor);
	sigmaFactor = sigmaMultiplier*(crowdingCoefficient / cFactor);
	//	output.printMessage("sigmaFactor (calculated)", sigmaFactor);

	int randomNumber = getRandomInteger(0,100);
	//	output.printMessage("randomNumber:", randomNumber);
	if(randomNumber % 5 == 0){

		double multiplier = getRandomDouble(2.0,3.0);
		//		output.printMessage("multiplier:", multiplier);
		sigmaFactor = sigmaFactor*multiplier;
		//		output.printMessage("In this iteration sigmaFactor is increased: ", sigmaFactor);


	}



	if (sigmaFactor < sigmaFactorMin)
		sigmaFactor = sigmaFactorMin;

	if (sigmaFactor > sigmaFactorMax)
		sigmaFactor = sigmaFactorMax;

	//	output.printMessage("sigmaFactor", sigmaFactor);
	objFun.setSigmaFactor(sigmaFactor);
}


void Optimizer::printIterationNumber(void) const {
	// Use ostringstream for more efficient string construction
	std::ostringstream oss;

	// Create a string with 100 spaces
	std::string padding(50, '*');

	// Build the output string
	oss << "\n" << padding << " Iteration = " << outerIterationNumber << " " << padding;

	// Send the string to the logger
	printInfoToLog(oss.str());

}

void Optimizer::printCurrentDesignToLogFile(void) {

	string msgCurrentDesign = "\n" + currentBestDesign.generateOutputString();
	printInfoToLog(msgCurrentDesign);

}

void Optimizer::printGlobalOptimalDesignToLogFile(void) {

	string msgCurrentDesign = globalOptimalDesign.generateOutputString();
	printInfoToLog("\n" + msgCurrentDesign);

}

void Optimizer::printHistory(void) const {
	history.print();
}

mat Optimizer::getHistory(void) const {
	return history.getData();
}




int Optimizer::findClosestDiscreteValue(double value, const vec& discreteValues) const {
	// Find the index of the closest discrete value
	int closestIndex = 0;
	double minDistance = fabs(value - discreteValues(0));

	for (unsigned int i = 1; i < discreteValues.getSize(); ++i) {
		double distance = fabs(value - discreteValues(i));
		if (distance < minDistance) {
			minDistance = distance;
			closestIndex = i;
		}
	}

	return closestIndex;
}

vec Optimizer::generateDiscreteValues(unsigned int index, double dx) const {
	unsigned int numDiscreteValues = static_cast<unsigned int>((upperBounds(index) - lowerBounds(index)) / dx) + 1;
	vec discreteValues(numDiscreteValues);

	for (unsigned int k = 0; k < numDiscreteValues; ++k) {
		discreteValues(k) = lowerBounds(index) + k * dx;
	}

	return discreteValues;
}


void Optimizer::roundDiscreteParameters(vec& designVector) const {
	if (numberOfDiscreteVariables > 0) {
		for (unsigned int j = 0; j < numberOfDiscreteVariables; ++j) {
			unsigned int index = indicesForDiscreteVariables[j];
			double valueToRound = designVector(index);
			double dx = incrementsForDiscreteVariables[j];

			// Generate discrete values for the current index
			vec discreteValues = generateDiscreteValues(index, dx);

			// Find the closest discrete value
			int closestIndex = findClosestDiscreteValue(valueToRound, discreteValues);
			designVector(index) = discreteValues(closestIndex);
		}
	}
}


void Optimizer::calculateImprovementValue(Design &d) {

	const double tolerance = 1e-5;
	// Set initial improvement value to zero
	d.improvementValue = 0.0;

	// Only calculate improvement if the design is feasible
	if (d.isDesignFeasible) {

		// If no feasible design has been found yet, initialize with the current design's value
		if (std::fabs(bestFeasibleInitialValue + std::numeric_limits<double>::max()) < tolerance) {
			bestFeasibleInitialValue = d.trueValue;
			// Assign a small positive improvement for the first feasible design
			d.improvementValue = std::numeric_limits<double>::epsilon();
		}

		// If the current design is better (smaller true value) than the best feasible design, calculate the improvement
		if (d.trueValue < bestFeasibleInitialValue) {
			d.improvementValue = bestFeasibleInitialValue - d.trueValue;
		}
	}
}


double Optimizer::normalCdf(double value, double mu, double sigma) const {
	if (sigma < 0.0) {
		throw std::invalid_argument("Standard deviation cannot be less than 0.");
	}

	// Handle the case where sigma is 0 or very close to zero
	if (sigma < 1e-12) {
		// If sigma is very small, treat it as effectively zero
		return (value < mu) ? 0.0 : 1.0;
	}

	// Calculate Z-score and CDF using the error function for normal distribution
	double z = (value - mu) / (sigma * SQRT2 );
	return 0.5 * (1 + std::erf(z));
}

double Optimizer::calculateProbabilityLessThanAValue(double value, double mu, double sigma) const {
	// Probability that a normal variable is less than a certain value
	return normalCdf(value, mu, sigma);
}

double Optimizer::calculateProbabilityGreaterThanAValue(double value, double mu, double sigma) const {
	// Probability that a normal variable is greater than a certain value
	return 1.0 - normalCdf(value, mu, sigma);
}




} /* Namespace Rodop */


