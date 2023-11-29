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


#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
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

void Optimizer::setDimension(unsigned int dim){

	dimension = dim;
	sampleDim = dimension;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	initializeBoundsForAcquisitionFunctionMaximization();
	iterMaxAcquisitionFunction = dimension*10000;

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

void Optimizer::setMinimumNumberOfSamplesAfterZoomIn(unsigned int nSamples){

	minimumNumberOfSamplesAfterZoomIn = nSamples;

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

void Optimizer::setMaxSigmaFactor(double value){
	assert(value>0.0);
	maximumSigma = value;

}
void Optimizer::setMinSigmaFactor(double value){
	assert(value>0.0);
	minimumSigma = value;

}


void Optimizer::setZoomInOff(void){
	ifZoomInDesignSpaceIsAllowed = false;
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


}



void Optimizer::addPenaltyToAcqusitionFunctionForConstraints(DesignForBayesianOptimization &designCalculated) const{

	if(isConstrained()){

		estimateConstraints(designCalculated);

		calculateFeasibilityProbabilities(designCalculated);

		designCalculated.updateAcqusitionFunctionAccordingToConstraints();

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


bool Optimizer::isToZoomInIteration(unsigned int iterationNo) const{

	if(iterationNo%howOftenZoomIn == 0 && ifZoomInDesignSpaceIsAllowed){

		return true;
	}
	else{
		return false;
	}

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

void Optimizer::modifyBoundsForInnerIterations(void){

	output.printMessage("modifying inner iteration bounds...");

	assert(zoomFactorShrinkageRate < 1.0 && zoomFactorShrinkageRate > 0.0);
	assert(dimension>0);
	assert(zoomFactor>0);

	findTheGlobalOptimalDesign();

	vec dvGlobalOptimum = trans(globalOptimalDesign.designParametersNormalized);

	vec deltaFromGlobalOptimimToLowerBounds = dvGlobalOptimum - 0.0;
	vec deltaFromGlobalOptimimToUpperBounds = 1.0/dimension - dvGlobalOptimum;

	if(IfSchrinkBounds){
		zoomFactor = zoomFactor*zoomFactorShrinkageRate;
	}
	if(IfEnlargeBounds){
		double oneOverShrinkageRate = 1.0/zoomFactorShrinkageRate;
		zoomFactor = zoomFactor* oneOverShrinkageRate;
	}

	output.printMessage("Factor for zoom in = ", zoomFactor);
	output.printMessage("Max Factor for zoom in = ", maxValueForZoomFactor);
	output.printMessage("Min Factor for zoom in = ", maxValueForZoomFactor);

	if(zoomFactor > maxValueForZoomFactor){

		IfSchrinkBounds = true;
		IfEnlargeBounds = false;
		maxValueForZoomFactor = maxValueForZoomFactor*0.5;
		if(maxValueForZoomFactor < 0.1){
			maxValueForZoomFactor = 0.1;
		}

	}
	if(zoomFactor < minValueForZoomFactor ){

		IfSchrinkBounds = false;
		IfEnlargeBounds = true;
	}


	vec newDeltasToLowerBound = deltaFromGlobalOptimimToLowerBounds*zoomFactor;
	vec newDeltasToUpperBound = deltaFromGlobalOptimimToUpperBounds*zoomFactor;


	lowerBoundsForAcqusitionFunctionMaximization = dvGlobalOptimum - newDeltasToLowerBound;
	upperBoundsForAcqusitionFunctionMaximization = dvGlobalOptimum + newDeltasToUpperBound;

	for(unsigned int i=0; i<dimension; i++){

		if(lowerBoundsForAcqusitionFunctionMaximization(i) < 0.0){
			lowerBoundsForAcqusitionFunctionMaximization(i) = 0;
		}
		if(upperBoundsForAcqusitionFunctionMaximization(i) > (1.0/dimension) ){
			upperBoundsForAcqusitionFunctionMaximization(i) = 1.0/dimension;
		}
	}



//	output.printMessage("lower bounds for inner iterations", lowerBoundsForAcqusitionFunctionMaximization);
//	output.printMessage("upper bounds for inner iterations", upperBoundsForAcqusitionFunctionMaximization);

}

void Optimizer::findTheGlobalOptimalDesign(void){

	assert(optimizationHistory.n_rows > 0);
	assert(ifBoxConstraintsSet);

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

	rowvec dvNormalized = normalizeVector(dv,lowerBounds,upperBounds);
	globalOptimalDesign.designParametersNormalized = dvNormalized;

	output.printDesign(globalOptimalDesign);

	globalOptimalDesign.saveToAFile(globalOptimumDesignFileName);

}




/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void Optimizer::findTheMostPromisingDesign(void){

	output.printMessage("Searching the best potential design...");

	double bestFeasibleObjectiveFunctionValue = globalOptimalDesign.trueValue;
	output.printMessage("Best feasible objective value = ", bestFeasibleObjectiveFunctionValue);

	objFun.setFeasibleMinimum(bestFeasibleObjectiveFunctionValue);


	assert(ifSurrogatesAreInitialized);
	assert(globalOptimalDesign.designParametersNormalized.size() > 0);

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;

	theMostPromisingDesigns.clear();


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
		assert(ID>=0 && ID < int(numberOfConstraints) );
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

void Optimizer::setNumberOfThread(unsigned int n){

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


void Optimizer::performEfficientGlobalOptimization(void){

	assert(ifObjectFunctionIsSpecied);
	assert(ifBoxConstraintsSet);

	checkIfSettingsAreOK();

	if(output.ifScreenDisplay){
		print();
	}

	initializeSurrogates();

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

		if(outerIterationNumber%howOftenTrainModels == 1) {

			initializeSurrogates();
			trainSurrogates();
		}


		if(isToZoomInIteration(outerIterationNumber)){

			modifyBoundsForInnerIterations();
		}

		findTheMostPromisingDesign();

		DesignForBayesianOptimization optimizedDesignGradientBased = MaximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));
		//		DesignForBayesianOptimization optimizedDesignGradientBased = theMostPromisingDesigns.at(0);

#if 0
		optimizedDesignGradientBased.print();
#endif




		rowvec best_dvNorm = optimizedDesignGradientBased.dv;
		rowvec best_dv =normalizeVectorBack(best_dvNorm, lowerBounds, upperBounds);
		double estimatedBestdv = objFun.interpolate(best_dvNorm);


		if(output.ifScreenDisplay){

			std::cout<<"The most promising design (not normalized):\n";
			best_dv.print();
			std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";
			std::cout<<"Variance = "<<optimizedDesignGradientBased.sigma<<"\n";
			cout<<"Acquisition function = " << optimizedDesignGradientBased.valueAcqusitionFunction << "\n";

		}



		if(areDiscreteParametersUsed()){
			roundDiscreteParameters(best_dv);
		}

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

		if(output.ifScreenDisplay){
			currentBestDesign.print();
		}

		findTheGlobalOptimalDesign();


		SURROGATE_MODEL type = objFun.getSurrogateModelType();

		if (type == GRADIENT_ENHANCED) {

			std::cout<<"currentBestDesign = "<<currentBestDesign.improvementValue<<"\n";
			std::cout<<"globalOptimalDesign = "<<globalOptimalDesign.improvementValue<<"\n";

			if(currentBestDesign.improvementValue <= globalOptimalDesign.improvementValue){

				std::cout<<"adding with zero gradients\n";
				objFun.setDataAddMode("adjointWithZeroGradient");
			}

		}


		objFun.addDesignToData(currentBestDesign);
		addConstraintValuesToData(currentBestDesign);

		updateOptimizationHistory(currentBestDesign);

		if(ifAdaptSigmaFactor){
			modifySigmaFactor();
		}

		/* terminate optimization */
		if(outerIterationNumber >= maxNumberOfSamples){


			output.printMessage("number of simulations > maximum number of simulations! Optimization is terminating...");
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

	std::cout<<"modifySigmaFactor ... \n";

	assert(minimumSigma<= maximumSigma);

	double b = maximumSigma;
	double a = (minimumSigma - maximumSigma)/(0.5*maxNumberOfSamples);


	if(outerIterationNumber < maxNumberOfSamples/2){

		sigmaFactor = a*outerIterationNumber + b;
	}
	else{

		sigmaFactor = minimumSigma;
	}

	//	objFun.setSigmaFactor(sigmaFactor);
	//	printScalar(sigmaFactor);

}




