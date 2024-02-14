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
#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>


#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../Auxiliary/INCLUDE/print.hpp"
#include "../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"
#include "../INCLUDE/Rodeo_globals.hpp"
#include "../TestFunctions/INCLUDE/test_functions.hpp"
#include "./INCLUDE/optimization.hpp"
#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

Optimizer::Optimizer(){}


Optimizer::Optimizer(std::string nameTestcase, int numberOfOptimizationParams){

	assert(isNotEmpty(nameTestcase));
	assert(numberOfOptimizationParams>0);

	name = nameTestcase;
	setDimension(numberOfOptimizationParams);

}

void Optimizer::initializeBoundsForAcquisitionFunctionMaximization() {

	lowerBoundsForAcqusitionFunctionMaximization.zeros(dimension);
	upperBoundsForAcqusitionFunctionMaximization.zeros(dimension);
	upperBoundsForAcqusitionFunctionMaximization.fill(1.0 / dimension);
}

void Optimizer::initializeOptimizerSettings(void) {

	factorForGradientStepWindow = 0.01;
	maximumIterationGradientStep = 5;
	numberOfThreads  = 1;
	sigmaFactor = 1.0;
	iterGradientEILoop = 100;
}


void Optimizer::setDimension(unsigned int dim){

	dimension = dim;
	sampleDim = dimension;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	initializeBoundsForAcquisitionFunctionMaximization();
	iterMaxAcquisitionFunction = dimension*10000;

	globalOptimalDesign.setDimension(dim);
	currentBestDesign.setDimension(dim);

}

void Optimizer::setName(std::string problemName){
	name = problemName;
}

void Optimizer::setParameterToDiscrete(unsigned int index, double increment){

	assert(index <dimension);
	incrementsForDiscreteVariables.push_back(increment);
	indicesForDiscreteVariables.push_back(index);
	numberOfDisceteVariables++;


}

bool Optimizer::checkSettings(void) const{

	output.printMessage("Checking settings...");

	bool ifAllSettingsOk = true;

	if(!ifBoxConstraintsSet){

		ifAllSettingsOk = false;

	}

	return ifAllSettingsOk;
}


void Optimizer::setInitialImprovementValue(double value){

	initialImprovementValue = value;
	IfinitialValueForObjFunIsSet = true;

}

void Optimizer::setMaximumNumberOfIterations(unsigned int maxIterations){

	maxNumberOfSamples = maxIterations;

}


void Optimizer::setMaximumNumberOfIterationsLowFidelity(unsigned int maxIterations){

	maxNumberOfSamplesLowFidelity =  maxIterations;


}



void Optimizer::setMaximumNumberOfInnerIterations(unsigned int maxIterations){

	iterMaxAcquisitionFunction = maxIterations;

}


void Optimizer::setFileNameDesignVector(std::string filename){

	assert(!filename.empty());
	designVectorFileName = filename;

}


void Optimizer::setBoxConstraints(Bounds boxConstraints){

	lowerBounds = boxConstraints.getLowerBounds();
	upperBounds = boxConstraints.getUpperBounds();


	assert(ifObjectFunctionIsSpecied);
	objFun.setParameterBounds(boxConstraints);

	if(isConstrained()){

		for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

			it->setParameterBounds(boxConstraints);
		}

	}

	globalOptimalDesign.setBoxConstraints(boxConstraints);
	ifBoxConstraintsSet = true;
}


void Optimizer::setDisplayOn(void){
	output.ifScreenDisplay = true;
}
void Optimizer::setDisplayOff(void){
	output.ifScreenDisplay = false;
}



void Optimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;
	globalOptimalDesign.setNumberOfConstraints(numberOfConstraints);
	sampleDim++;

}


void Optimizer::addObjectFunction(ObjectiveFunction &objFunc){

	assert(ifObjectFunctionIsSpecied == false);

	objFun = objFunc;

	designVectorFileName = objFun.getFileNameDesignVector();

	objFun.setSigmaFactor(sigmaFactor);

	sampleDim++;
	ifObjectFunctionIsSpecied = true;

}


void Optimizer::evaluateConstraints(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->setEvaluationMode("primal");
		it->evaluateDesign(d);
	}
}



void Optimizer::estimateConstraints(DesignForBayesianOptimization &design) const{

	rowvec x = design.dv;
	assert(design.constraintValues.size() == numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::pair<double, double> result = it->interpolateWithVariance(x);

		design.constraintValues(it->getID()) = result.first;
		design.constraintSigmas(it->getID()) = result.second;

	}
}


bool Optimizer::checkBoxConstraints(void) const{

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) return false;
	}

	return true;
}



bool Optimizer::checkConstraintFeasibility(rowvec constraintValues) const{

	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		bool ifFeasible = it->checkFeasibility(constraintValues(i));

		if(ifFeasible == false) {
			return false;
		}
		i++;
	}

	return true;
}


void Optimizer::calculateFeasibilityProbabilities(DesignForBayesianOptimization &designCalculated) const{

	rowvec probabilities(numberOfConstraints, fill::zeros);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		string type  = it->getInequalityType();
		double inequalityValue = it->getInequalityTargetValue();
		int ID = it->getID();
		double estimated = designCalculated.constraintValues(ID);
		double sigma = designCalculated.constraintSigmas(ID);

		if(type.compare(">") == 0){
			/* p (constraint value > target) */
			probabilities(ID) =  calculateProbalityGreaterThanAValue(inequalityValue, estimated, sigma);
		}

		if(type.compare("<") == 0){
			probabilities(ID) =  calculateProbalityLessThanAValue(inequalityValue, estimated, sigma);
		}

	}

	designCalculated.constraintFeasibilityProbabilities = probabilities;

}



void Optimizer::print(void) const{

	std::cout<<"\nOptimizer Settings = \n\n";
	std::cout<<"Problem name : "<<name<<"\n";
	std::cout<<"Dimension    : "<<dimension<<"\n";
	std::cout<<"Maximum number of function evaluations: " <<maxNumberOfSamples<<"\n";
	std::cout<<"Maximum number of iterations for EI maximization: " << iterMaxAcquisitionFunction <<"\n";


	objFun.print();

	if (isNotConstrained()){
		std::cout << "Optimization problem does not have any constraints\n";
	}
	else{

		printConstraints();
	}

	if(numberOfDisceteVariables > 0){
		std::cout << "Indices for discrete parameters = \n";
		printVector(indicesForDiscreteVariables);
		std::cout << "Incremental values for discrete parameters = \n";
		printVector(incrementsForDiscreteVariables);
	}

}

void Optimizer::printConstraints(void) const{

	std::cout<< "List of constraints = \n";

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->print();
	}

}


void Optimizer::initializeSurrogates(void){

	assert(ifObjectFunctionIsSpecied);

	output.printMessage("Initializing surrogate model for the objective function...");

	objFun.initializeSurrogate();

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		output.printMessage("Initializing surrogate models for the constraint...");

		it->initializeSurrogate();

	}

	ifSurrogatesAreInitialized = true;

	output.printMessage("Initialization is done...");

}

void Optimizer::trainSurrogatesForConstraints() {
	output.printMessage("Training surrogate model for the constraints...");
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end();
			it++) {
		it->trainSurrogate();
	}
	output.printMessage("Model training for constraints is done...");
}

void Optimizer::trainSurrogates(void){

	output.printMessage("Training surrogate model for the objective function...");
	objFun.trainSurrogate();

	if(isConstrained()){
		trainSurrogatesForConstraints();
	}
}




void Optimizer::updateOptimizationHistory(Design d) {

	rowvec newSample(sampleDim+2);

	for(unsigned int i=0; i<dimension; i++) {
		newSample(i) = d.designParameters(i);
	}

	newSample(dimension) = d.trueValue;

	for(unsigned int i=0; i<numberOfConstraints; i++){
		newSample(i+dimension+1) = 	d.constraintTrueValues(i);
	}

	newSample(sampleDim)   = d.improvementValue;

	if(d.isDesignFeasible){
		newSample(sampleDim+1) = 1.0;
	}
	else{
		newSample(sampleDim+1) = 0.0;
	}

	optimizationHistory.insert_rows( optimizationHistory.n_rows, newSample );
	appendRowVectorToCSVData(newSample,"optimizationHistory.csv");

#if 0
	printf("optimizationHistory:\n");
	optimizationHistory.print();

#endif

	findTheGlobalOptimalDesign();



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


bool Optimizer::isConstrained(void) const{

	if(numberOfConstraints > 0) return true;
	else return false;
}

bool Optimizer::isNotConstrained(void) const{

	if(numberOfConstraints == 0) return true;
	else return false;
}

bool Optimizer::areDiscreteParametersUsed(void) const{

	if(numberOfDisceteVariables>0) return true;
	else return false;
}

void Optimizer::computeConstraintsandPenaltyTerm(Design &d) {

	if(isConstrained()){

		output.printMessage("Evaluating constraints...");

		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);
		if(!ifConstraintsSatisfied){

			output.printMessage("The new sample does not satisfy all the constraints");
			d.isDesignFeasible = false;

		}
		else{

			output.printMessage("The new sample satisfies all the constraints");
			d.isDesignFeasible = true;
		}

		output.printMessage("Evaluation of the constraints is ready...");
	}

}

void Optimizer::addConstraintValuesToData(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->addDesignToData(d);
	}

}

void Optimizer::checkIfSettingsAreOK(void) const{

	if (maxNumberOfSamples == 0){
		abortWithErrorMessage("Maximum number of samples is not set for the optimization");
	}
	if(checkBoxConstraints() == false){
		abortWithErrorMessage("Box constraints are not set properly");
	}


}


void Optimizer::findTheGlobalOptimalDesign(void){

	assert(optimizationHistory.n_rows > 0);
	assert(ifBoxConstraintsSet);

	globalOptimalDesign.setGlobalOptimalDesignFromHistoryFile(optimizationHistory);
	globalOptimalDesign.saveToXMLFile();

}


void Optimizer::changeSettingsForAGradientBasedStep(void){

	assert(factorForGradientStepWindow >  0.0);
	assert(factorForGradientStepWindow <= 1.0);
	assert(dimension>0);
	assert(globalOptimalDesign.designParametersNormalized.size() == dimension);

	vec dv = trans(globalOptimalDesign.designParametersNormalized);

	/* we divide by dimension since all 1/dimension is always a factor in normalizetion */
	double delta = factorForGradientStepWindow/dimension;

	lowerBoundsForAcqusitionFunctionMaximizationGradientStep = dv -  delta;
	upperBoundsForAcqusitionFunctionMaximizationGradientStep = dv +  delta;

	trimVectorSoThatItStaysWithinTheBounds(lowerBoundsForAcqusitionFunctionMaximizationGradientStep);
	trimVectorSoThatItStaysWithinTheBounds(upperBoundsForAcqusitionFunctionMaximizationGradientStep);


}

bool Optimizer::checkIfDesignIsWithinBounds(const rowvec &x0) const{

	assert(x0.size() == dimension);
	assert(lowerBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);
	assert(upperBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);

	vec x0T = trans(x0);
	vec lb = lowerBoundsForAcqusitionFunctionMaximizationGradientStep;
	vec ub = upperBoundsForAcqusitionFunctionMaximizationGradientStep;
	return isBetween(x0T,lb,ub);


}


bool Optimizer::checkIfDesignTouchesBounds(const rowvec &x0) const{

	assert(x0.size() == dimension);
	assert(lowerBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);
	assert(upperBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);

	double eps = 10E-6;
	vec a1 = lowerBoundsForAcqusitionFunctionMaximizationGradientStep - eps;
	vec b1 = lowerBoundsForAcqusitionFunctionMaximizationGradientStep + eps;

	vec a2 = upperBoundsForAcqusitionFunctionMaximizationGradientStep - eps;
	vec b2 = upperBoundsForAcqusitionFunctionMaximizationGradientStep + eps;

	for(unsigned int i=0;i<dimension; i++){

		if(isNumberBetween(x0(i), a1(i), b1(i))) {
			return true;
		}
		if(isNumberBetween(x0(i), a2(i), b2(i))) {
			return true;
		}
	}
	return false;
}


double Optimizer::determineMaxStepSizeForGradientStep(rowvec x0, rowvec gradient) const{

	assert(dimension>0);
	assert(x0.size() > 0);
	assert(x0.size() == gradient.size());
	assert(upperBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);
	assert(lowerBoundsForAcqusitionFunctionMaximizationGradientStep.size() == dimension);

	double stepSizeMax = 0;

	vec ub = upperBoundsForAcqusitionFunctionMaximizationGradientStep;
	vec lb = lowerBoundsForAcqusitionFunctionMaximizationGradientStep;

	vec stepSizeTemp(dimension, fill::zeros);

	for(unsigned int i=0; i<dimension; i++){

		/* x_k+1 = x_k - step * grad
		 *
		 * => -(x_k+1 - x_k)/grad =  step
		 *
		 */

		double step = 0.0;
		if(gradient(i) > 0.0){ /* x(i) will decrease so at minimum it will touch lb*/
			step = -(lb(i)  - x0(i))/gradient(i);
		}
		else{ /* else it will increase so at maximum it will touch ub */
			step = -(ub(i)  - x0(i))/gradient(i);
		}

		stepSizeTemp(i) = step;
	}

	stepSizeMax = stepSizeTemp.max();
	//	printScalar(stepSizeMax);

	return stepSizeMax;

}

void Optimizer::trimVectorSoThatItStaysWithinTheBounds(rowvec &x) {
	assert(dimension>0);
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

void Optimizer::trimVectorSoThatItStaysWithinTheBounds(vec &x) {
	assert(dimension>0);
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

void Optimizer::findPromisingDesignUnconstrainedGradientStep(
		DesignForBayesianOptimization &designToBeTried) {

	assert(dimension>0);

	rowvec gradient = globalOptimalDesign.gradient;
	rowvec x0 = globalOptimalDesign.designParametersNormalized;
	assert(gradient.size() == dimension);
	assert(x0.size() == dimension);

	//			gradient.print("gradient");
	output.printMessage("Staring point for the gradient step x0 = \n",x0);


	double maxStepSize = determineMaxStepSizeForGradientStep(x0, gradient);
	double minStepSize = (1.0/dimension)/10000;

	maxStepSize = maxStepSize * trustRegionFactorGradientStep;

	if(maxStepSize < 10.0*minStepSize){

		maxStepSize = 10.0* minStepSize;
	}


	output.printMessage("Maximum step size",maxStepSize);
	output.printMessage("Minimum step size",minStepSize);

	double deltaStepSize = (maxStepSize - minStepSize) / iterMaxAcquisitionFunction;
	double stepSize = minStepSize;
	/*
	 printScalar(trustRegionFactorGradientStep);
	 printScalar(maxStepSize);
	 printScalar(stepSize);
	 */

	double bestObjectiveFunctionValue = LARGE;
	rowvec bestDesignNormalized;
	for (unsigned int i = 0; i < iterMaxAcquisitionFunction; i++) {
		rowvec x = x0 - stepSize * gradient;

		trimVectorSoThatItStaysWithinTheBounds(x);
		//						x.print("x");
		designToBeTried.dv = x;
		objFun.calculateSurrogateEstimateUsingDerivatives(designToBeTried);
		double surrogateEstimate = designToBeTried.objectiveFunctionValue;
		//						printScalar(surrogateEstimate);
		if (surrogateEstimate < bestObjectiveFunctionValue) {
			bestObjectiveFunctionValue = surrogateEstimate;
			bestDesignNormalized = x;
			//			printScalar(bestObjectiveFunctionValue);
		}
		stepSize += deltaStepSize;
	}

	output.printMessage("Best design (inner iteration) = \n", bestDesignNormalized);
	output.printMessage("At the step size (inner iteration) = \n", stepSize);

	designToBeTried.dv = bestDesignNormalized;
}

void Optimizer::decideIfNextStepWouldBeAGradientStep() {
	if (iterationNumberGradientStep == maximumIterationGradientStep) {
		trustRegionFactorGradientStep = 1.0;
		iterationNumberGradientStep = 0;
		WillGradientStepBePerformed = false;
	}
}

void Optimizer::findTheMostPromisingDesignGradientStep(void){

	globalOptimalDesign.setGradientGlobalOptimumFromTrainingData(objFun.getFileNameTrainingData());

	globalOptimalDesign.setGradientGlobalOptimumFromTrainingData(objFun.getFileNameTrainingData());

	bool ifGradientVectorExists = globalOptimalDesign.checkIfGlobalOptimaHasGradientVector();

	if(!ifGradientVectorExists){
		WillGradientStepBePerformed = false;
	}



	if(WillGradientStepBePerformed){

		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);


		changeSettingsForAGradientBasedStep();

		if(isConstrained()){


			PRINT_HERE
			abort();

		}

		else{

			findPromisingDesignUnconstrainedGradientStep(designToBeTried);
		}

		theMostPromisingDesigns.push_back(designToBeTried);
		iterationNumberGradientStep++;
		trustRegionFactorGradientStep = trustRegionFactorGradientStep * 0.5;

		decideIfNextStepWouldBeAGradientStep();



	}

}


/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void Optimizer::findTheMostPromisingDesignEGO(void){

	output.printMessage("Searching the best potential design...");

	double bestFeasibleObjectiveFunctionValue = globalOptimalDesign.trueValue;
	output.printMessage("Best feasible objective value = ", bestFeasibleObjectiveFunctionValue);

	objFun.setFeasibleMinimum(bestFeasibleObjectiveFunctionValue);


	assert(ifSurrogatesAreInitialized);
	assert(globalOptimalDesign.designParametersNormalized.size() > 0);

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;

	DesignForBayesianOptimization designWithMaxAcqusition(dimension,numberOfConstraints);
	designWithMaxAcqusition.valueAcqusitionFunction = -LARGE;

#ifdef OPENMP_SUPPORT
	omp_set_num_threads(numberOfThreads);
#endif

#ifdef OPENMP_SUPPORT
#pragma omp parallel for
#endif
	for(unsigned int i = 0; i <iterMaxAcquisitionFunction; i++ ){


		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVector(lb, ub);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);

		if(designToBeTried.valueAcqusitionFunction > designWithMaxAcqusition.valueAcqusitionFunction){
#ifdef OPENMP_SUPPORT
#pragma omp critical
#endif
			{
				designWithMaxAcqusition = designToBeTried;

			}
#if 0
			printf("A design with a better EI value has been found\n");
			designToBeTried.print();
#endif
		}

	}

	//	designWithMaxEI.print();

	rowvec dvNotNormalized = normalizeVectorBack(globalOptimalDesign.designParametersNormalized, lowerBounds, upperBounds);
	if(checkIfBoxConstraintsAreSatisfied(dvNotNormalized)){

#ifdef OPENMP_SUPPORT
#pragma omp parallel for
#endif
		for(unsigned int i = 0; i < iterMaxAcquisitionFunction; i++ ){


			DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);

			rowvec dv0 = globalOptimalDesign.designParametersNormalized;
			designToBeTried.generateRandomDesignVectorAroundASample(dv0, lb, ub);



			objFun.calculateExpectedImprovement(designToBeTried);
			addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);


#if 0
			designToBeTried.print();
#endif
			if(designToBeTried.valueAcqusitionFunction > designWithMaxAcqusition.valueAcqusitionFunction){
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

	theMostPromisingDesigns.push_back(designWithMaxAcqusition);

}


DesignForBayesianOptimization Optimizer::getDesignWithMaxExpectedImprovement(void) const{
	return theMostPromisingDesigns.front();
}


rowvec Optimizer::calculateGradientOfAcqusitionFunction(DesignForBayesianOptimization &currentDesign) const{


	rowvec gradient(dimension);


	for(unsigned int i=0; i<dimension; i++){
#if 0
		printf("dv:\n");
		dvGradientSearch.print();
#endif

		double dvSave = currentDesign.dv(i);
#if 0
		printf("epsilon_EI = %15.10f\n",epsilon_EI);
#endif

		double epsilon = currentDesign.dv(i)*0.00001;
		currentDesign.dv(i) += epsilon;

#if 0
		printf("dv perturbed:\n");
		dvPerturbed.print();
#endif

		objFun.calculateExpectedImprovement(currentDesign);

		double EIplus = currentDesign.valueAcqusitionFunction;
		currentDesign.dv(i) -= 2*epsilon;

		objFun.calculateExpectedImprovement(currentDesign);

		double EIminus = currentDesign.valueAcqusitionFunction;


		/* obtain the forward finite difference quotient */
		double fdVal = (EIplus - EIminus)/(2*epsilon);
		gradient(i) = fdVal;
		currentDesign.dv(i) = dvSave;


	} /* end of finite difference loop */
#if 0
	printf("Gradient vector:\n");
	gradEI.print();
#endif

	return gradient;
}



DesignForBayesianOptimization Optimizer::MaximizeAcqusitionFunctionGradientBased(DesignForBayesianOptimization initialDesign) const {

	rowvec gradEI(dimension);
	double stepSize0 = 0.001;
	double stepSize = 0.0;


	objFun.calculateExpectedImprovement(initialDesign);
	addPenaltyToAcqusitionFunctionForConstraints(initialDesign);


	double EI0 = initialDesign.valueAcqusitionFunction;
	DesignForBayesianOptimization bestDesign = initialDesign;

	bool breakOptimization = false;

	for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){

#if 0
		printf("\nGradient search iteration = %d\n", iterGradientSearch);
#endif


		gradEI = calculateGradientOfAcqusitionFunction(bestDesign);

		/* save the design vector */
		DesignForBayesianOptimization dvLineSearchSave = bestDesign ;

#if 0
		printf("Line search...\n");
#endif

		stepSize = stepSize0;

		while(1){


			/* design update */

			vec lb = lowerBoundsForAcqusitionFunctionMaximization;
			vec ub = upperBoundsForAcqusitionFunctionMaximization;

			bestDesign.gradientUpdateDesignVector(gradEI,lb,ub,stepSize);

			objFun.calculateExpectedImprovement(bestDesign);
			addPenaltyToAcqusitionFunctionForConstraints(bestDesign);


#if 0
			printf("EI_LS = %15.10f\n",bestDesign.valueExpectedImprovement );

#endif

			/* if ascent is achieved */
			if(bestDesign.valueAcqusitionFunction > EI0){
#if 0
				printf("Ascent is achieved with difference = %15.10f\n", bestDesign.valueExpectedImprovement -  EI0);

				bestDesign.print();
#endif
				EI0 = bestDesign.valueAcqusitionFunction;
				break;
			}

			else{ /* else halve the stepsize and set design to initial */

				stepSize = stepSize * 0.5;
				bestDesign = dvLineSearchSave;
#if 0
				printf("stepsize = %15.10f\n",stepSize);

#endif
				if(stepSize < 10E-12) {
#if 0
					printf("The stepsize is getting too small!\n");
#endif

					breakOptimization = true;
					break;
				}
			}

		}

		if(breakOptimization) break;

	} /* end of gradient-search loop */


	return bestDesign;


}
void Optimizer::prepareOptimizationHistoryFile(void) const{

	std::string header;
	for(unsigned int i=0; i<dimension; i++){

		header+="x";
		header+=std::to_string(i+1);
		header+=",";

	}

	header+="Objective Function,";


	for(unsigned int i=0; i<numberOfConstraints; i++){
		header+="Constraint";
		header+=std::to_string(i+1);

		header+=",";

	}

	header+="Improvement,";
	header+="Feasibility";
	header+="\n";

	std::ofstream optimizationHistoryFile;
	optimizationHistoryFile.open (optimizationHistoryFileName);
	optimizationHistoryFile << header;
	optimizationHistoryFile.close();



}

void Optimizer::clearOptimizationHistoryFile(void) const{

	remove(optimizationHistoryFileName.c_str());

}


void Optimizer::setHowOftenTrainModels(unsigned int value){
	howOftenTrainModels = value;
}

void Optimizer::setOptimizationHistoryConstraints(mat inputObjectiveFunction) {

	unsigned int N = inputObjectiveFunction.n_rows;

	for (auto it = constraintFunctions.begin();it != constraintFunctions.end(); it++) {

		int ID = it->getID();
		assert(ID>=0 && ID < int(numberOfConstraints) );
		mat dataRead;
		string fileNameConstraint = it->getFileNameTrainingData();
		dataRead.load(fileNameConstraint, csv_ascii);

		mat inputConstraint = dataRead.submat(0, 0, dataRead.n_rows - 1,
				dimension - 1);

		string type = it->getInequalityType();

		for (unsigned int i = 0; i < N; i++) {
			assert(i<inputObjectiveFunction.size());
			rowvec input = inputObjectiveFunction.row(i);
			int indx = findIndexOfRow(input, inputConstraint, 10E-8);
			if (indx >= 0) {
				optimizationHistory(i, dimension + ID + 1) = dataRead(indx,
						dimension);
			} else {

				if (isEqual(type, ">")) {
					optimizationHistory(i, dimension + ID + 1) = -LARGE;
				}
				if (isEqual(type, "<")) {
					optimizationHistory(i, dimension + ID + 1) = LARGE;
				}
			}
		}
	}
}

bool Optimizer::checkIfBoxConstraintsAreSatisfied(const rowvec &designVariables) const{

	assert(ifBoxConstraintsSet);
	assert(designVariables.size() > 0);
	assert(designVariables.size() == dimension);
	assert(designVariables.size() == lowerBounds.size());

	bool isDesignFeasible = true;
	for (unsigned int i=0; i < dimension; i++) {
		if (designVariables(i) < lowerBounds(i) || designVariables(i) > upperBounds(i)) {
			isDesignFeasible = false;
		}
	}
	return isDesignFeasible;
}

void Optimizer::setOptimizationHistoryFeasibilityValues(void){

	assert(optimizationHistory.n_rows > 0);
	assert(dimension>0);

	unsigned int N = optimizationHistory.n_rows;
	for (unsigned int i = 0; i < N; i++) {

		rowvec rowOfTheHistoryFile = optimizationHistory.row(i);
		bool isFeasible = true;

		if(isConstrained()){

			rowvec constraintValues(numberOfConstraints);

			for(unsigned int j=0; j<numberOfConstraints; j++){

				constraintValues(j) = rowOfTheHistoryFile(j+dimension+1);
			}
			isFeasible = checkConstraintFeasibility(constraintValues);
		}

		rowvec dv = rowOfTheHistoryFile.head(dimension);
		bool ifBoxConstraintsAreSatisfied = checkIfBoxConstraintsAreSatisfied(dv);

		unsigned int nCols = optimizationHistory.n_cols;

		if(isFeasible && ifBoxConstraintsAreSatisfied) {

			optimizationHistory(i,nCols-1) = 1.0;
		}
		else{

			optimizationHistory(i,nCols-1) = 0.0;
		}

	}

}

void Optimizer::setNumberOfThreads(unsigned int n){
	numberOfThreads = n;
}

void Optimizer::calculateInitialImprovementValue(void){

	unsigned int N = optimizationHistory.n_rows;
	assert(N>0);
	unsigned int nCols = optimizationHistory.n_cols;
	unsigned int indexLastCol = nCols - 1;


	bool ifFeasibleDesignFound = false;

	double bestFeasibleObjectiveFunctionValue = LARGE;


	vec objectiveFunctionValues = optimizationHistory.col(dimension);

	for(unsigned int i=0; i<N; i++){

		double feasibility = optimizationHistory(i,indexLastCol);

		if(feasibility>0.0 && objectiveFunctionValues(i) < bestFeasibleObjectiveFunctionValue){
			ifFeasibleDesignFound = true;
			bestFeasibleObjectiveFunctionValue = objectiveFunctionValues(i);

		}
	}

	if(ifFeasibleDesignFound){
		setInitialImprovementValue(bestFeasibleObjectiveFunctionValue);
	}

}

void Optimizer::setOptimizationHistory(void){

	assert(ifSurrogatesAreInitialized);

	string filenameObjFun = objFun.getFileNameTrainingData();

	mat trainingDataObjectiveFunction;

	trainingDataObjectiveFunction.load(filenameObjFun, csv_ascii);
	unsigned int N = trainingDataObjectiveFunction.n_rows;

	mat inputObjectiveFunction = trainingDataObjectiveFunction.submat(0, 0, N-1, dimension-1 );

	optimizationHistory.reset();
	optimizationHistory = zeros<mat>(N,dimension + numberOfConstraints +3);

	for(unsigned int i=0; i<dimension; i++){
		optimizationHistory.col(i) = inputObjectiveFunction.col(i);
	}

	optimizationHistory.col(dimension) = trainingDataObjectiveFunction.col(dimension);

	if(isConstrained()){
		setOptimizationHistoryConstraints(inputObjectiveFunction);
	}

	setOptimizationHistoryFeasibilityValues();

	appendMatrixToCSVData(optimizationHistory,"optimizationHistory.csv");

	calculateInitialImprovementValue();

	findTheGlobalOptimalDesign();

#if 0
	globalOptimalDesign.print();
#endif
}



void Optimizer::evaluateObjectiveFunctionMultiFidelityWithBothPrimal(Design &currentBestDesign) {
	objFun.setEvaluationMode("primal");
	objFun.evaluateDesign(currentBestDesign);
	objFun.setEvaluationMode("primalLowFi");
	objFun.evaluateDesign(currentBestDesign);
	objFun.setDataAddMode("primalBoth");

}

void Optimizer::evaluateObjectiveFunctionMultiFidelityWithLowFiAdjoint(Design &currentBestDesign) {

	objFun.setEvaluationMode("primal");
	objFun.evaluateDesign(currentBestDesign);
	objFun.setEvaluationMode("adjointLowFi");
	objFun.evaluateDesign(currentBestDesign);
	objFun.setDataAddMode("primalHiFiAdjointLowFi");

}

void Optimizer::evaluateObjectiveFunctionMultiFidelity(Design &currentBestDesign) {

	SURROGATE_MODEL typeHiFi = objFun.getSurrogateModelType();
	SURROGATE_MODEL typeLowFi = objFun.getSurrogateModelTypeLowFi();

	if (typeHiFi == ORDINARY_KRIGING && typeLowFi == ORDINARY_KRIGING) {

		evaluateObjectiveFunctionMultiFidelityWithBothPrimal(currentBestDesign);
	}
	if (typeHiFi == ORDINARY_KRIGING && typeLowFi == GRADIENT_ENHANCED) {
		evaluateObjectiveFunctionMultiFidelityWithLowFiAdjoint(currentBestDesign);
	}
}

void Optimizer::evaluateObjectiveFunctionWithTangents(Design &currentBestDesign) {

	currentBestDesign.generateRandomDifferentiationDirection();
	objFun.setEvaluationMode("tangent");
	objFun.setDataAddMode("tangent");
}

void Optimizer::evaluateObjectFunctionWithAdjoints() {
	objFun.setEvaluationMode("adjoint");
	objFun.setDataAddMode("adjoint");
}

void Optimizer::evaluateObjectFunctionWithPrimal() {
	objFun.setEvaluationMode("primal");
	objFun.setDataAddMode("primal");
}

void Optimizer::evaluateObjectiveFunctionSingleFidelity(Design &currentBestDesign) {

	SURROGATE_MODEL type = objFun.getSurrogateModelType();

	if (type == TANGENT_ENHANCED) {
		evaluateObjectiveFunctionWithTangents(currentBestDesign);
	} else if (type == GRADIENT_ENHANCED) {
		evaluateObjectFunctionWithAdjoints();
	} else {
		evaluateObjectFunctionWithPrimal();
	}

	objFun.evaluateDesign(currentBestDesign);

}

void Optimizer::evaluateObjectiveFunction(Design &currentBestDesign) {
	/* now make a simulation for the most promising design */


	if(objFun.isMultiFidelityActive()){

		evaluateObjectiveFunctionMultiFidelity(currentBestDesign);
	}

	else{

		evaluateObjectiveFunctionSingleFidelity(currentBestDesign);
	}

}

void Optimizer::setDataAddModeForGradientBasedStep(
		const Design &currentBestDesign) {
	if (currentBestDesign.improvementValue
			<= globalOptimalDesign.improvementValue) {
		output.printMessage(
				"Addding the new sample without gradient since no descent has been achieved");
		objFun.setDataAddMode("adjointWithZeroGradient");
	} else {
		output.printMessage(
				"Addding the new sample with gradient since descent has been achieved");
		WillGradientStepBePerformed = true;
		trustRegionFactorGradientStep = 1.0;
		iterationNumberGradientStep = 0;
	}
}

void Optimizer::findTheMostPromisingDesignToBeSimulated() {

	DesignForBayesianOptimization optimizedDesignGradientBased;

	theMostPromisingDesigns.clear();

	if (doesObjectiveFunctionHaveGradients() && WillGradientStepBePerformed) {
		output.printMessage("A Gradient Step will be performed...");
		output.printMessage("Line search iteration number = ", iterationNumberGradientStep);
		findTheMostPromisingDesignGradientStep();

		optimizedDesignGradientBased = theMostPromisingDesigns.at(0);


	} else {
		output.printMessage("An EGO Step will be performed...");
		findTheMostPromisingDesignEGO();
		optimizedDesignGradientBased = MaximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));
	}

	rowvec best_dvNorm = optimizedDesignGradientBased.dv;
	rowvec best_dv =normalizeVectorBack(best_dvNorm, lowerBounds, upperBounds);



	if(output.ifScreenDisplay){

		std::cout<<"The most promising design (not normalized):\n";
		best_dv.print();
		double estimatedBestdv = objFun.interpolate(best_dvNorm);
		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";
		std::cout<<"Variance = "<<optimizedDesignGradientBased.sigma<<"\n";
		cout<<"Acquisition function = " << optimizedDesignGradientBased.valueAcqusitionFunction << "\n";

	}


	if(areDiscreteParametersUsed()){
		roundDiscreteParameters(best_dv);
	}

	currentBestDesign.designParametersNormalized = best_dvNorm;
	currentBestDesign.designParameters = best_dv;

}

void Optimizer::initializeCurrentBestDesign(void) {
	currentBestDesign.tag = "The most promising design";
	currentBestDesign.setNumberOfConstraints(numberOfConstraints);
	currentBestDesign.saveDesignVector(designVectorFileName);
	currentBestDesign.isDesignFeasible = true;
}

void Optimizer::abortIfCurrentDesignHasANaN() {
	if (currentBestDesign.checkIfHasNan()) {
		abortWithErrorMessage("NaN while reading external executable outputs");
	}
}

void Optimizer::decideIfAGradientStepShouldBeTakenForTheFirstIteration() {

	assert(isHistoryFileInitialized);
	findTheGlobalOptimalDesign();

	if (doesObjectiveFunctionHaveGradients()) {
		globalOptimalDesign.setGradientGlobalOptimumFromTrainingData(objFun.getFileNameTrainingData());

		if (globalOptimalDesign.checkIfGlobalOptimaHasGradientVector()) {
			WillGradientStepBePerformed = true;
		}
	}
}

void Optimizer::performEfficientGlobalOptimization(void){

	assert(ifObjectFunctionIsSpecied);
	assert(ifBoxConstraintsSet);

	checkIfSettingsAreOK();

	initializeOptimizerSettings();

	if(output.ifScreenDisplay){
		print();
	}

	initializeSurrogates();
	initializeCurrentBestDesign();

	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
		setOptimizationHistory();
		isHistoryFileInitialized = true;
	}

	/* main loop for optimization */

	while(1){

		outerIterationNumber++;

		output.printIteration(outerIterationNumber);



		if(outerIterationNumber == 1){
			decideIfAGradientStepShouldBeTakenForTheFirstIteration();
		}

		if(outerIterationNumber%howOftenTrainModels == 1) {
			initializeSurrogates();
			trainSurrogates();
		}

		findTheMostPromisingDesignToBeSimulated();

		DesignForBayesianOptimization optimizedDesignGradientBased = MaximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));

#if 0
		optimizedDesignGradientBased.print();
#endif

		/* now make a simulation for the most promising design */
		evaluateObjectiveFunction(currentBestDesign);
		computeConstraintsandPenaltyTerm(currentBestDesign);

		calculateImprovementValue(currentBestDesign);
		printScalar(currentBestDesign.improvementValue);




		abortIfCurrentDesignHasANaN();

		output.printDesign(currentBestDesign);

		if (doesObjectiveFunctionHaveGradients()) {
			setDataAddModeForGradientBasedStep(currentBestDesign);
		}


		objFun.addDesignToData(currentBestDesign);
		addConstraintValuesToData(currentBestDesign);


		if(currentBestDesign.improvementValue > globalOptimalDesign.improvementValue){

			double deltaImprovement = currentBestDesign.improvementValue - globalOptimalDesign.improvementValue;

			if(deltaImprovement > bestDeltaImprovementValueAchieved){

				bestDeltaImprovementValueAchieved = deltaImprovement;
			}

			output.printMessage("An IMPROVEMENT has been achieved!");

			double percentImprovementRelativeToBest = (deltaImprovement/ bestDeltaImprovementValueAchieved)*100;
			printScalar(percentImprovementRelativeToBest);

			if(percentImprovementRelativeToBest < 5){

				WillGradientStepBePerformed = false;
				trustRegionFactorGradientStep = 1.0;
				iterationNumberGradientStep = 0;
			}

		}



		updateOptimizationHistory(currentBestDesign);


		/* terminate optimization */
		if(outerIterationNumber >= maxNumberOfSamples){

			output.printMessage("number of simulations > maximum number of simulations! Optimization is terminating...");
			output.printDesign(globalOptimalDesign);
			break;
		}

		if(doesObjectiveFunctionHaveGradients()){
			objFun.removeVeryCloseSamples(globalOptimalDesign);
		}



	} /* end of the optimization loop */

}



void Optimizer::roundDiscreteParameters(rowvec &designVector){

	if(numberOfDisceteVariables>0){

		for(unsigned int j=0; j < numberOfDisceteVariables; j++){


			unsigned int index = indicesForDiscreteVariables[j];
			double valueToRound = designVector(index);

			double dx = incrementsForDiscreteVariables[j];
			unsigned int howManyDiscreteValues = (upperBounds(index) - lowerBounds(index))/dx;
			howManyDiscreteValues += 1;

			vec discreteValues(howManyDiscreteValues);

			discreteValues(0) = lowerBounds(index);
			for(unsigned int k=1; k<howManyDiscreteValues-1; k++){

				discreteValues(k) = discreteValues(k-1) + dx;


			}

			discreteValues(howManyDiscreteValues-1) = upperBounds(index);

			int whichInterval = findInterval(valueToRound, discreteValues);


			assert(whichInterval>=0);

			double distance1 = valueToRound - discreteValues[whichInterval];
			double distance2 = discreteValues[whichInterval+1] - valueToRound;

			if(distance1 < distance2){

				designVector(index) =  discreteValues[whichInterval];

			}
			else{

				designVector(index) =  discreteValues[whichInterval+1];
			}


		}

	}


}


void Optimizer::calculateImprovementValue(Design &d){

	if(d.isDesignFeasible){

		if(!IfinitialValueForObjFunIsSet){

			initialImprovementValue  = d.trueValue;
			IfinitialValueForObjFunIsSet = true;
		}

		if(d.trueValue < initialImprovementValue){
			d.improvementValue = initialImprovementValue - d.trueValue;
		}

	}
}




