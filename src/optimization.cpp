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
#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxiliary_functions.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "lhs.hpp"
#include "vector_manipulations.hpp"
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

void Optimizer::setDimension(unsigned int dim){

	dimension = dim;
	sampleDim = dimension;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	initializeBoundsForAcquisitionFunctionMaximization();
	iterMaxAcquisitionFunction = dimension*100000;

	minDeltaXForZoom = 0.01/dimension;
	globalOptimalDesign.setDimension(dim);
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

	if(ifConstrained()){

		for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

			it->setParameterBounds(boxConstraints);
		}

	}


	ifBoxConstraintsSet = true;
}


void Optimizer::setDisplayOn(void){
	output.ifScreenDisplay = true;
}
void Optimizer::setDisplayOff(void){
	output.ifScreenDisplay = false;
}

void Optimizer::setZoomInOn(void){
	ifZoomInDesignSpaceIsAllowed = true;
}

void Optimizer::setZoomFactor(double value){
	assert(value>0.0);
	assert(value<1.0);

	zoomFactorShrinkageRate = value;

}



void Optimizer::setZoomInOff(void){
	ifZoomInDesignSpaceIsAllowed = false;
}


pair<vec,vec> Optimizer::getBoundsForAcqusitionFunctionMaximization(void) const{
	pair<vec,vec> result;
	result.first  = lowerBoundsForAcqusitionFunctionMaximization;
	result.second = upperBoundsForAcqusitionFunctionMaximization;

	return result;
}

Design Optimizer::getGlobalOptimalDesign(void) const{
	return globalOptimalDesign;
}

void Optimizer::setHowOftenZoomIn(unsigned int value){
	assert(value < maxNumberOfSamples);
	howOftenZoomIn = value;
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

	bool flagWithinBounds = true;

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) flagWithinBounds = false;
	}

	return flagWithinBounds;
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

	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		string type  = it->getInequalityType();
		double value = it->getInequalityTargetValue();
		int ID = it->getID();
		double estimated = designCalculated.constraintValues(ID);
		double sigma = designCalculated.constraintSigmas(ID);

		if(type.compare(">") == 0){
			/* p (constraint value > target) */
			probabilities(ID) =  calculateProbalityGreaterThanAValue(value, estimated, sigma);
		}

		if(type.compare("<") == 0){
			probabilities(ID) =  calculateProbalityLessThanAValue(value, estimated, sigma);
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

	if (!ifConstrained()){
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


void Optimizer::trainSurrogates(void){

	output.printMessage("Training surrogate model for the objective function...");

	objFun.trainSurrogate();

	if(constraintFunctions.size() !=0){
		output.printMessage("Training surrogate model for the constraints...");
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->trainSurrogate();
	}

	if(constraintFunctions.size() !=0){
		output.printMessage("Model training for constraints is done...");
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


}



void Optimizer::addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &designCalculated) const{

	if(ifConstrained()){

		estimateConstraints(designCalculated);

		calculateFeasibilityProbabilities(designCalculated);

		designCalculated.updateAcqusitionFunctionAccordingToConstraints();

	}
}


bool Optimizer::ifConstrained(void) const{

	if(numberOfConstraints > 0) return true;
	else return false;

}


void Optimizer::computeConstraintsandPenaltyTerm(Design &d) {

	if(ifConstrained()){

		output.printMessage("Evaluating constraints...");

		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);
		if(!ifConstraintsSatisfied){

			output.printMessage("The new sample does not satisfy all the constraints");
			d.isDesignFeasible = false;

		}

		output.printMessage("Evaluation of the constraints is ready...");
	}
	else{

		d.isDesignFeasible = true;
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

void Optimizer::zoomInDesignSpace(void){

	output.printMessage("Zooming in design space...");

	findTheGlobalOptimalDesign();

#if 0
	globalOptimalDesign.print();
#endif
	vec dx = upperBoundsForAcqusitionFunctionMaximization - lowerBoundsForAcqusitionFunctionMaximization;

	rowvec dvNormalized = normalizeVector(globalOptimalDesign.designParameters, lowerBounds, upperBounds);

#if 0
	printVector(dvNormalized,"dvNormalized");
#endif

	for(unsigned int i=0; i<dimension; i++){

		double delta = dx(i)*zoomInFactor;

		if(delta < minDeltaXForZoom){

			delta = minDeltaXForZoom;
		}

		lowerBoundsForAcqusitionFunctionMaximization(i) =  dvNormalized(i) - delta;
		upperBoundsForAcqusitionFunctionMaximization(i) =  dvNormalized(i) + delta;

		if(lowerBoundsForAcqusitionFunctionMaximization(i) < 0.0) {
			lowerBoundsForAcqusitionFunctionMaximization(i) = 0.0;
		}

		if(upperBoundsForAcqusitionFunctionMaximization(i) > 1.0/dimension) {
			upperBoundsForAcqusitionFunctionMaximization(i) = 1.0/dimension;

		}

	}

	output.printMessage("New bounds for the acquisition function are set...");
	output.printBoxConstraints(lowerBoundsForAcqusitionFunctionMaximization,upperBoundsForAcqusitionFunctionMaximization);

	zoomInFactor = zoomInFactor * zoomFactorShrinkageRate;
}


void Optimizer::reduceBoxConstraints(void){

	output.printMessage("Reducing box constraints...");

	vec lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec ub = upperBoundsForAcqusitionFunctionMaximization;

	vec lbNotNormalized = normalizeVectorBack(lb, lowerBounds, upperBounds);
	vec ubNotNormalized = normalizeVectorBack(ub, lowerBounds, upperBounds);


	output.printMessage("Updated box constraints = ");

	output.printBoxConstraints(lbNotNormalized,ubNotNormalized);

	Bounds newBoxConstraints(lbNotNormalized,ubNotNormalized);

	setBoxConstraints(newBoxConstraints);


}



void Optimizer::reduceTrainingDataFiles(void) const{

	assert(ifBoxConstraintsSet);

	output.printMessage("Reducing size of training data...");

	objFun.reduceTrainingDataFiles(howManySamplesReduceAfterZoomIn, globalOptimalDesign.trueValue);

	if(ifConstrained()){

		for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

			double targetValue = it->getInequalityTargetValue();
			it->reduceTrainingDataFiles(howManySamplesReduceAfterZoomIn,targetValue );
		}
	}

}

void Optimizer::findTheGlobalOptimalDesign(void){

	assert(optimizationHistory.n_rows > 0);

	unsigned int indexLastCol = optimizationHistory.n_cols -1;

	bool isFeasibleDesignFound = false;
	double bestObjectiveFunctionValue = LARGE;
	unsigned int bestIndex;

	for(unsigned int i=0; i<optimizationHistory.n_rows; i++){

		double feasibility = optimizationHistory(i,indexLastCol);
		double objectiveFunctionValue = optimizationHistory(i,dimension);

		if(feasibility>0.0 && objectiveFunctionValue < bestObjectiveFunctionValue){
			isFeasibleDesignFound = true;
			bestObjectiveFunctionValue = objectiveFunctionValue;
			bestIndex = i;
		}

	}

	rowvec bestSample;
	if(isFeasibleDesignFound){

		bestSample = optimizationHistory.row(bestIndex);

		globalOptimalDesign.isDesignFeasible = true;
		globalOptimalDesign.ID = bestIndex;
	}

	else{

		vec objectiveFunctionValues = optimizationHistory.col(dimension);

		uword indexMin = index_min(objectiveFunctionValues);
		bestSample = optimizationHistory.row(indexMin);

		globalOptimalDesign.isDesignFeasible = false;
		globalOptimalDesign.ID = indexMin;
	}

	rowvec dv = bestSample.head(dimension);

	globalOptimalDesign.tag = "Global optimum design";
	globalOptimalDesign.designParameters  = dv;
	globalOptimalDesign.trueValue = bestSample(dimension);
	globalOptimalDesign.improvementValue = bestSample(optimizationHistory.n_cols-2);

	rowvec constraintValues(numberOfConstraints);
	for(unsigned int i=0; i<numberOfConstraints; i++){

		constraintValues(i) = bestSample(i+dimension+1);
	}

	globalOptimalDesign.constraintTrueValues = constraintValues;

	output.printDesign(globalOptimalDesign);

	globalOptimalDesign.saveToAFile(globalOptimumDesignFileName);

}




/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void Optimizer::findTheMostPromisingDesign(unsigned int howManyDesigns){

	output.printMessage("Searching the best potential design...");

	assert(ifSurrogatesAreInitialized);

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;

	theMostPromisingDesigns.clear();


	DesignForBayesianOptimization designWithMaxAcqusition(dimension,numberOfConstraints);
	designWithMaxAcqusition.valueAcqusitionFunction = -LARGE;

#pragma omp parallel for
	for(unsigned int i = 0; i <iterMaxAcquisitionFunction; i++ ){


		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);



		designToBeTried.generateRandomDesignVector(lb, ub);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);

		if(designToBeTried.valueAcqusitionFunction > designWithMaxAcqusition.valueAcqusitionFunction){

			designWithMaxAcqusition = designToBeTried;
#if 0
			printf("A design with a better EI value has been found\n");
			designToBeTried.print();
#endif
		}

	}

	//	designWithMaxEI.print();



#pragma omp parallel for
	for(unsigned int i = 0; i < iterMaxAcquisitionFunction; i++ ){


		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVectorAroundASample(designWithMaxAcqusition.dv, lb, ub);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueAcqusitionFunction > designWithMaxAcqusition.valueAcqusitionFunction){

			designWithMaxAcqusition = designToBeTried;
#if 0
			printf("A design with a better EI value has been found (second loop) \n");
			designToBeTried.print();
#endif
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

			bestDesign.gradientUpdateDesignVector(gradEI,lowerBoundsForAcqusitionFunctionMaximization,upperBoundsForAcqusitionFunctionMaximization,stepSize);

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
		assert(ID>=0 && ID<numberOfConstraints );
		mat dataRead;
		string fileNameConstraint = it->getFileNameTrainingData();
		dataRead.load(fileNameConstraint, csv_ascii);

		mat inputConstraint = dataRead.submat(0, 0, dataRead.n_rows - 1,
				dimension - 1);

		string type = it->getInequalityType();

		for (unsigned int i = 0; i < N; i++) {
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

void Optimizer::setOptimizationHistoryFeasibilityValues(mat inputObjectiveFunction){

	unsigned int N = inputObjectiveFunction.n_rows;
	for (unsigned int i = 0; i < N; i++) {
		rowvec rowOfHistory = optimizationHistory.row(i);
		rowvec constraintValues(numberOfConstraints);

		for(unsigned int j=0; j<numberOfConstraints; j++){

			constraintValues(j) = rowOfHistory(j+dimension+1);
		}

		bool isFeasible = checkConstraintFeasibility(constraintValues);

		unsigned int nCols = optimizationHistory.n_cols;

		if(isFeasible) {

			optimizationHistory(i,nCols-1) = 1.0;
		}
		else{

			optimizationHistory(i,nCols-1) = 0.0;
		}

	}

}

void Optimizer::calculateInitialImprovementValue(void){

	unsigned int N = optimizationHistory.n_rows;
	assert(N>0);
	unsigned int nCols = optimizationHistory.n_cols;
	unsigned int indexLastCol = nCols - 1;


	bool ifFeasibleDesignFound = false;

	double bestFeasibleObjectiveFunctionValue = LARGE;
	int bestIndex = -1;

	vec objectiveFunctionValues = optimizationHistory.col(dimension);

	for(unsigned int i=0; i<N; i++){

		double feasibility = optimizationHistory(i,indexLastCol);

		if(feasibility>0.0 && objectiveFunctionValues(i) < bestFeasibleObjectiveFunctionValue){
			ifFeasibleDesignFound = true;
			bestFeasibleObjectiveFunctionValue = objectiveFunctionValues(i);
			bestIndex = i;
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

	if (ifConstrained()){
		setOptimizationHistoryConstraints(inputObjectiveFunction);
		setOptimizationHistoryFeasibilityValues(inputObjectiveFunction);
	}
	else{

		vec ones(N);
		ones.fill(1.0);
		optimizationHistory.col(optimizationHistory.n_cols -1) = ones;
	}

	appendMatrixToCSVData(optimizationHistory,"optimizationHistory.csv");

	calculateInitialImprovementValue();

	findTheGlobalOptimalDesign();

#if 0
	globalOptimalDesign.print();
#endif
}


mat Optimizer::getOptimizationHistory(void) const{
	return optimizationHistory;
}

void Optimizer::evaluateObjectiveFunction(Design &currentBestDesign) {
	/* now make a simulation for the most promising design */


	if(objFun.isMultiFidelityActive()){

		SURROGATE_MODEL typeHiFi = objFun.getSurrogateModelType();
		SURROGATE_MODEL typeLowFi = objFun.getSurrogateModelTypeLowFi();

		if(typeHiFi == ORDINARY_KRIGING && typeLowFi == ORDINARY_KRIGING){

			objFun.setEvaluationMode("primal");
			objFun.evaluateDesign(currentBestDesign);
			objFun.setEvaluationMode("primalLowFi");
			objFun.evaluateDesign(currentBestDesign);
			objFun.setDataAddMode("primalBoth");

			objFun.addDesignToData(currentBestDesign);

		}

		if(typeHiFi == ORDINARY_KRIGING && typeLowFi == GRADIENT_ENHANCED){

			objFun.setEvaluationMode("primal");
			objFun.evaluateDesign(currentBestDesign);
			objFun.setEvaluationMode("adjointLowFi");
			objFun.evaluateDesign(currentBestDesign);
			objFun.setDataAddMode("primalHiFiAdjointLowFi");

			objFun.addDesignToData(currentBestDesign);


		}

	}

	else{

		SURROGATE_MODEL type = objFun.getSurrogateModelType();
		if(type == TANGENT){

			currentBestDesign.generateRandomDifferentiationDirection();
			objFun.setEvaluationMode("tangent");
			objFun.setDataAddMode("tangent");

		}
		else if(type == GRADIENT_ENHANCED){

			objFun.setEvaluationMode("adjoint");
			objFun.setDataAddMode("adjoint");
		}

		else{

			objFun.setEvaluationMode("primal");
			objFun.setDataAddMode("primal");

		}


		objFun.evaluateDesign(currentBestDesign);
		objFun.addDesignToData(currentBestDesign);

	}

}

void Optimizer::EfficientGlobalOptimization(void){

	assert(ifObjectFunctionIsSpecied);
	assert(ifBoxConstraintsSet);

	checkIfSettingsAreOK();

	print();

	output.ifScreenDisplay = true;

	initializeSurrogates();

	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
		setOptimizationHistory();
		isHistoryFileInitialized = true;
	}

	/* main loop for optimization */
	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;

	while(1){

		iterOpt++;

		output.printIteration(iterOpt);

		if(simulationCount%howOftenTrainModels == 0) {

			initializeSurrogates();
			trainSurrogates();
		}

		if(iterOpt%howOftenZoomIn == 0){

			if(ifZoomInDesignSpaceIsAllowed) {

				zoomInDesignSpace();
				reduceBoxConstraints();
				reduceTrainingDataFiles();
				initializeSurrogates();
				trainSurrogates();

			}

		}

		findTheMostPromisingDesign();

		DesignForBayesianOptimization optimizedDesignGradientBased = MaximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));

#if 0
		optimizedDesignGradientBased.print();
#endif


		rowvec best_dvNorm = optimizedDesignGradientBased.dv;
		rowvec best_dv =normalizeVectorBack(best_dvNorm, lowerBounds, upperBounds);
		double estimatedBestdv = objFun.interpolate(best_dvNorm);


#if 1
		printf("The most promising design (not normalized):\n");
		best_dv.print();
		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";
		std::cout<<"Sigma = "<<optimizedDesignGradientBased.sigma<<"\n";
		cout<<"Acquisition function = " << optimizedDesignGradientBased.valueAcqusitionFunction << "\n";

#endif


		roundDiscreteParameters(best_dv);

		Design currentBestDesign(best_dv);
		currentBestDesign.tag = "The most promising design";
		currentBestDesign.setNumberOfConstraints(numberOfConstraints);
		currentBestDesign.saveDesignVector(designVectorFileName);
		currentBestDesign.isDesignFeasible = true;

		/* now make a simulation for the most promising design */

		evaluateObjectiveFunction(currentBestDesign);


		computeConstraintsandPenaltyTerm(currentBestDesign);

		calculateImprovementValue(currentBestDesign);

		if(currentBestDesign.checkIfHasNan()){
			abortWithErrorMessage("NaN while reading external executable outputs");
		}

		currentBestDesign.print();

		addConstraintValuesToData(currentBestDesign);
		updateOptimizationHistory(currentBestDesign);

		findTheGlobalOptimalDesign();

		if(ifAdaptSigmaFactor){
			modifySigmaFactor();
		}


		simulationCount ++;

		/* terminate optimization */
		if(simulationCount >= maxNumberOfSamples){


			output.printMessage("number of simulations > max_number_of_samples! Optimization is terminating...");
			output.printDesign(globalOptimalDesign);
			break;
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

void Optimizer::modifySigmaFactor(void){

	vec improvementValues = optimizationHistory.col(sampleDim);

//	improvementValues.print("improvement values");
    uword i = improvementValues.index_max();

    if(i == improvementValues.size()-1){


    	std::cout<<"Improvement has been achieved in the last iteration\n";
    	if(sigmaFactor>2.0) sigmaFactor = 2.0;
    	sigmaFactor = sigmaFactor*0.8;
    }
    else{

    	std::cout<<"No improvement has been achieved in the last iteration\n";
    	sigmaFactor = sigmaFactor*1.25;
    }

    if(sigmaFactor > 10) sigmaFactor = 10.0;
    if(sigmaFactor < 0.5) sigmaFactor = 0.5;


    objFun.setSigmaFactor(sigmaFactor);
    printScalar(sigmaFactor);

}




