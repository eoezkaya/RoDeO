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
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

Optimizer::Optimizer(){}


Optimizer::Optimizer(std::string nameTestcase, int numberOfOptimizationParams){

	name = nameTestcase;
	dimension = numberOfOptimizationParams;
	sampleDim = dimension;

	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);

	initializeBoundsForAcquisitionFunctionMaximization();

	iterMaxAcqusitionFunction = dimension*10000;


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
	iterMaxAcqusitionFunction = dimension*10000;

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

	if(this->ifDisplay){

		std::cout<<"Checking settings...\n";

	}


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



void Optimizer::setMaximumNumberOfIterationsForEIMaximization(unsigned int maxIterations){

	iterMaxAcqusitionFunction = maxIterations;

}


void Optimizer::setFileNameDesignVector(std::string filename){

	assert(!filename.empty());
	designVectorFileName = filename;

}


void Optimizer::setBoxConstraints(std::string filename){

	assert(!filename.empty());

	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;

	}

	mat boxConstraints;

	bool status = boxConstraints.load(filename.c_str(), csv_ascii);
	if(status == true)
	{
		std::cout<<"Input for the box constraints is done"<<std::endl;
	}
	else
	{
		std::cout<<"Problem with data the input (cvs ascii format) at"<<__FILE__<<", line:"<<__LINE__<<std::endl;
		exit(-1);
	}

	for(unsigned int i=0; i<dimension; i++){

		assert(boxConstraints(i,0) < boxConstraints(i,1));

	}

	lowerBounds = boxConstraints.col(0);
	upperBounds = boxConstraints.col(1);
	ifBoxConstraintsSet = true;


}

void Optimizer::setBoxConstraints(double lowerBound, double upperBound){

	assert(lowerBound < upperBound);
	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;
	}

	assert(lowerBound < upperBound);
	lowerBounds.fill(lowerBound);
	upperBounds.fill(upperBound);
	ifBoxConstraintsSet = true;

}


void Optimizer::setBoxConstraints(vec lb, vec ub){

	assert(lb.size()>0);
	assert(lb.size() == ub.size());
	assert(ifBoxConstraintsSet == false);

	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;

	}
	for(unsigned int i=0; i<dimension; i++) assert(lb(i) < ub(i));

	lowerBounds = lb;
	upperBounds = ub;
	ifBoxConstraintsSet = true;

}


void Optimizer::setBoxConstraints(Bounds boxConstraints){

	lowerBounds = boxConstraints.getLowerBounds();
	upperBounds = boxConstraints.getUpperBounds();
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

	sampleDim++;
	ifObjectFunctionIsSpecied = true;

}





void Optimizer::evaluateConstraints(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		if(!it->checkIfGradientAvailable()){

			it->setEvaluationMode("primal");
			it->evaluateDesign(d);

		}else{

			abort();

		}

	}
}

void Optimizer::addConstraintValuesToDoEData(Design &d) const{

	unsigned int countConstraintWithGradient = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::string filenameCVS = it->getName()+".csv";

		if(ifDisplay){

			std::cout<<"Appending to data: "<<filenameCVS<<"\n";

		}

		if(it->checkIfGradientAvailable()){


			rowvec saveBuffer(2*dimension+1);
			copyRowVector(saveBuffer,d.designParameters);
			saveBuffer(dimension) = d.constraintTrueValues(it->getID());

			rowvec gradient = d.constraintGradients[countConstraintWithGradient];
			countConstraintWithGradient++;
			copyRowVector(saveBuffer,gradient,dimension+1);

			appendRowVectorToCSVData(saveBuffer,filenameCVS);



		}else{


			rowvec saveBuffer(dimension+1);
			copyRowVector(saveBuffer,d.designParameters);
			saveBuffer(dimension) = d.constraintTrueValues(it->getID());
			appendRowVectorToCSVData(saveBuffer,filenameCVS);


		}


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
	std::cout<<"Maximum number of iterations for EI maximization: " <<iterMaxAcqusitionFunction <<"\n";


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


void Optimizer::visualizeOptimizationHistory(void) const{

	if(dimension == 2){

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_2d_opthist.py "+ name;
#if 0
		cout<<python_command<<"\n";
#endif
		FILE* in = popen(python_command.c_str(), "r");

		fprintf(in, "\n");

	}


}


void Optimizer::initializeSurrogates(void){

	assert(ifObjectFunctionIsSpecied);

	displayMessage("Initializing surrogate model for the objective function...\n");

	objFun.initializeSurrogate();

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		displayMessage("Initializing surrogate models for the constraint...\n");

		it->initializeSurrogate();

	}

	ifSurrogatesAreInitialized = true;

	displayMessage("Initialization is done...");

}


void Optimizer::trainSurrogates(void){

	displayMessage("Training surrogate model for the objective function...\n");

	objFun.trainSurrogate();

	if(constraintFunctions.size() !=0){
		displayMessage("Training surrogate model for the constraints...\n");
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->trainSurrogate();
	}

	if(constraintFunctions.size() !=0){
		displayMessage("Model training for constraints is done...");
	}
}




void Optimizer::updateOptimizationHistory(Design d) {

	rowvec newSample(sampleDim+1);

	for(unsigned int i=0; i<dimension; i++) {

		newSample(i) = d.designParameters(i);

	}

	newSample(dimension) = d.trueValue;


	for(unsigned int i=0; i<numberOfConstraints; i++){

		newSample(i+dimension+1) = 	d.constraintTrueValues(i);
	}
	newSample(sampleDim) = d.improvementValue;



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

		displayMessage("Evaluating constraints...\n");

		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);
		if(!ifConstraintsSatisfied){

			output.printMessage("The new sample does not satisfy all the constraints");
			d.isDesignFeasible = false;

		}
	}

	output.printMessage("Evaluation of the constraints is ready...");

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

	displayMessage("Zooming in design space...\n");

	findTheGlobalOptimalDesign();

#if 0
	globalOptimalDesign.print();
#endif
	vec dx = upperBoundsForAcqusitionFunctionMaximization - lowerBoundsForAcqusitionFunctionMaximization;

	rowvec dvNormalized = normalizeRowVector(globalOptimalDesign.designParameters, lowerBounds, upperBounds);

#if 1
	printVector(dvNormalized,"dvNormalized");
#endif

	for(unsigned int i=0; i<dimension; i++){

		double delta = dx(i)*zoomInFactor;

		if(delta < 10E-5){

			delta = 10E-5;
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
#if 0
	printVector(lowerBoundsForAcqusitionFunctionMaximization,"lowerBoundsForEIMaximization" );
	printVector(upperBoundsForAcqusitionFunctionMaximization,"upperBoundsForEIMaximization");
#endif

	zoomInFactor = zoomInFactor* zoomFactorShrinkageRate;
}



void Optimizer::findTheGlobalOptimalDesign(void){

	assert(optimizationHistory.n_rows > 0);

	unsigned int indexLastCol = optimizationHistory.n_cols -1;

	output.printMessage("Finding the global design...");

	bool isFeasibleDesignFound;
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
	globalOptimalDesign.saveToAFile(globalOptimumDesignFileName);

}

void Optimizer::findTheGlobalOptimalDesignMultiFidelity(void){

	output.printMessage("Finding the global design...");


	double bestObjectiveFunctionValue = LARGE;
	for (auto it = begin (highFidelityDesigns); it != end (highFidelityDesigns); ++it) {

		double J = it->trueValue;

		if(J < bestObjectiveFunctionValue && it->isDesignFeasible){

			bestObjectiveFunctionValue = J;
			globalOptimalDesign = *it;

		}

	}

}


/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void Optimizer::findTheMostPromisingDesign(unsigned int howManyDesigns){

	assert(ifSurrogatesAreInitialized);

	vec &lb = lowerBoundsForAcqusitionFunctionMaximization;
	vec &ub = upperBoundsForAcqusitionFunctionMaximization;

	theMostPromisingDesigns.clear();


	DesignForBayesianOptimization designWithMaxEI(dimension,numberOfConstraints);

#pragma omp parallel for
	for(unsigned int i = 0; i <iterMaxAcqusitionFunction; i++ ){


		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);



		designToBeTried.generateRandomDesignVector(lb, ub);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);

		//
		//		if(designToBeTried.valueAcqusitionFunction>0){
		//			designToBeTried.print();
		//		}


		if(designToBeTried.valueAcqusitionFunction > designWithMaxEI.valueAcqusitionFunction){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found\n");
			designToBeTried.print();
#endif
		}

	}

	//	designWithMaxEI.print();



#pragma omp parallel for
	for(unsigned int i = 0; i < iterMaxAcqusitionFunction; i++ ){


		DesignForBayesianOptimization designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVectorAroundASample(designWithMaxEI.dv, lb, ub);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToAcqusitionFunctionForConstraints(designToBeTried);

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueAcqusitionFunction > designWithMaxEI.valueAcqusitionFunction){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found (second loop) \n");
			designToBeTried.print();
#endif
		}

	}

	theMostPromisingDesigns.push_back(designWithMaxEI);

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

			bestDesign.gradientUpdateDesignVector(gradEI,stepSize);

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

	double bestObjectiveFunctionValue = LARGE;
	int bestIndex = -1;

	for(unsigned int i=0; i<N; i++){

		double feasibility = optimizationHistory(i,indexLastCol);
		double objectiveFunctionValue = optimizationHistory(i,dimension);

		if(feasibility>0.0 && objectiveFunctionValue < bestObjectiveFunctionValue){
			ifFeasibleDesignFound = true;
			bestObjectiveFunctionValue = objectiveFunctionValue;
			bestIndex = i;
		}
	}
	if(ifFeasibleDesignFound){
		initialImprovementValue = bestObjectiveFunctionValue;
	}




}

void Optimizer::setOptimizationHistory(void){

	assert(ifSurrogatesAreInitialized);

	string filenameObjFun = objFun.getFileNameTrainingData();

	mat trainingDataObjectiveFunction;

	trainingDataObjectiveFunction.load(filenameObjFun, csv_ascii);
	unsigned int N = trainingDataObjectiveFunction.n_rows;

	mat inputObjectiveFunction = trainingDataObjectiveFunction.submat(0, 0, N-1, dimension-1 );

	optimizationHistory = zeros<mat>(N,dimension + numberOfConstraints +3);

	for(unsigned int i=0; i<dimension; i++){
		optimizationHistory.col(i) = inputObjectiveFunction.col(i);
	}


	optimizationHistory.col(dimension) = trainingDataObjectiveFunction.col(dimension);

	if (ifConstrained()){
		setOptimizationHistoryConstraints(inputObjectiveFunction);
		setOptimizationHistoryFeasibilityValues(inputObjectiveFunction);
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


void Optimizer::EfficientGlobalOptimization(void){

	assert(ifObjectFunctionIsSpecied);
	assert(ifBoxConstraintsSet);

	checkIfSettingsAreOK();


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

		output.printMessage("############################################");
		output.printMessage("Iteration = ",iterOpt);

		if(simulationCount%howOftenTrainModels == 0) {

			trainSurrogates();
		}

		if(iterOpt%howOftenZoomIn == 0){

			if(ifZoomInDesignSpaceIsAllowed) zoomInDesignSpace();

		}

		findTheMostPromisingDesign();

		DesignForBayesianOptimization optimizedDesignGradientBased = MaximizeAcqusitionFunctionGradientBased(theMostPromisingDesigns.at(0));

#if 0
		optimizedDesignGradientBased.print();
#endif


		rowvec best_dvNorm = optimizedDesignGradientBased.dv;
		rowvec best_dv =normalizeRowVectorBack(best_dvNorm, lowerBounds, upperBounds);
		double estimatedBestdv = objFun.interpolate(best_dvNorm);

#if 0
		printf("The most promising design (not normalized):\n");
		best_dv.print();
		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";
#endif


		roundDiscreteParameters(best_dv);

		Design currentBestDesign(best_dv);
		currentBestDesign.tag = "Current best design";
		currentBestDesign.setNumberOfConstraints(numberOfConstraints);
		currentBestDesign.saveDesignVector(designVectorFileName);
		currentBestDesign.isDesignFeasible = true;


		/* now make a simulation for the most promising design */

		if(!objFun.checkIfGradientAvailable()) {

			objFun.setEvaluationMode("primal");
			objFun.evaluateDesign(currentBestDesign);
			objFun.addDesignToData(currentBestDesign);


		}
		else{

			objFun.setEvaluationMode("adjoint");
			objFun.evaluateDesign(currentBestDesign);
			objFun.addDesignToData(currentBestDesign);

		}


		computeConstraintsandPenaltyTerm(currentBestDesign);



		calculateImprovementValue(currentBestDesign);

		if(currentBestDesign.checkIfHasNan()){
			abortWithErrorMessage("NaN while reading external executable outputs");
		}
#if 0
		currentBestDesign.print();
#endif

		addConstraintValuesToData(currentBestDesign);
		updateOptimizationHistory(currentBestDesign);

		findTheGlobalOptimalDesign();

		if(ifDisplay){

			std::cout<<"##########################################\n";
			std::cout<<"Optimization Iteration = "<<iterOpt<<"\n";
			currentBestDesign.print();
			std::cout<<"\n\n";

		}
		simulationCount ++;

		/* terminate optimization */
		if(simulationCount >= maxNumberOfSamples){


			if(ifDisplay){

				printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");

				std::cout<<"##########################################\n";
				std::cout<<"Global best design = \n";
				globalOptimalDesign.print();
				std::cout<<"\n\n";

			}

			if(ifVisualize){

				visualizeOptimizationHistory();
			}

			break;
		}

	} /* end of the optimization loop */

}


//void Optimizer::EfficientGlobalOptimization2(void){
//
//
//	mat dataHighFidelity;
//
//	dataHighFidelity.load("HimmelblauHiFiData.csv", csv_ascii);
//
//	vec JHiFi = dataHighFidelity.col(2);
//
//
//
//	checkIfSettingsAreOK();
//
//
//	if(!isHistoryFileInitialized){
//
//		clearOptimizationHistoryFile();
//		prepareOptimizationHistoryFile();
//
//	}
//
//	/* main loop for optimization */
//	unsigned int simulationCount = 0;
//	unsigned int iterOpt=0;
//
//	initializeSurrogates();
//
//	while(1){
//
//		iterOpt++;
//#if 0
//		printf("Optimization Iteration = %d\n",iterOpt);
//#endif
//
//		if(simulationCount%howOftenTrainModels == 0) {
//
//			trainSurrogates();
//		}
//
//		//		if(iterOpt%10 == 0){
//		//
//		//			zoomInDesignSpace();
//		//
//		//		}
//
//		findTheMostPromisingDesign();
//
//		DesignForBayesianOptimization optimizedDesignGradientBased = MaximizeEIGradientBased(theMostPromisingDesigns.at(0));
//
//#if 0
//		optimizedDesignGradientBased.print();
//#endif
//
//
//		rowvec best_dvNorm = optimizedDesignGradientBased.dv;
//		rowvec best_dv =normalizeRowVectorBack(best_dvNorm, lowerBounds, upperBounds);
//		double estimatedBestdv = objFun.interpolate(best_dvNorm);
//
//#if 0
//		printf("The most promising design (not normalized):\n");
//		best_dv.print();
//		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";
//
//#endif
//
//
//		roundDiscreteParameters(best_dv);
//
//
//		Design currentBestDesign(best_dv);
//		currentBestDesign.setNumberOfConstraints(numberOfConstraints);
//		currentBestDesign.saveDesignVector(designVectorFileName);
//
//
//		/* now make a simulation for the most promising design */
//
//		if(!objFun.checkIfGradientAvailable()) {
//
//			//CEO			objFun.evaluateLowFidelity(currentBestDesign);
//
//		}
//		else{
//
//			//CEO			objFun.evaluateAdjoint(currentBestDesign);
//
//		}
//
//		//CEO		objFun.readEvaluateOutput(currentBestDesign);
//		objFun.addLowFidelityDesignToData(currentBestDesign);
//
//
//		std::cout<<"Low fidelity design\n";
//		currentBestDesign.print();
//
//
//		computeConstraintsandPenaltyTerm(currentBestDesign);
//
//		calculateImprovementValue(currentBestDesign);
//
//		if(currentBestDesign.checkIfHasNan()){
//
//			cout<<"ERROR: NaN while reading external executable outputs!\n";
//			abort();
//
//		}
//#if 0
//		currentBestDesign.print();
//#endif
//
//		addConstraintValuesToData(currentBestDesign);
//		lowFidelityDesigns.push_back(currentBestDesign);
//
//		rowvec v1 = currentBestDesign.designParameters;
//		rowvec v2 = normalizeRowVector(v1, lowerBounds,upperBounds);
//
//		double fv2 = objFun.interpolate(v2);
//		std::cout<<"HiFi estimate = "<<fv2<<"\n";
//
//		vec P = { 0.1, 0.50, 0.75 };
//
//		printVector(JHiFi);
//
//		vec Q = quantile(JHiFi, P);
//
//
//		Q.print("Q");
//
//
//
//		if( fv2  < Q(0)) {
//
//
//			//CEO			objFun.evaluate(currentBestDesign);
//			//CEO			objFun.readEvaluateOutput(currentBestDesign);
//			objFun.addDesignToData(currentBestDesign);
//
//			std::cout<<"High fidelity design\n";
//			currentBestDesign.print();
//
//			highFidelityDesigns.push_back(currentBestDesign);
//
//			addOneElement(JHiFi,currentBestDesign.trueValue );
//
//
//		}
//
//
//		findTheGlobalOptimalDesignMultiFidelity();
//
//
//		globalOptimalDesign.saveToAFile(globalOptimumDesignFileName);
//
//
//		simulationCount ++;
//
//		/* terminate optimization */
//		if(simulationCount >= maxNumberOfSamples){
//
//
//			if(ifDisplay){
//
//				printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");
//
//				std::cout<<"##########################################\n";
//				std::cout<<"Global best design = \n";
//				globalOptimalDesign.print();
//				std::cout<<"\n\n";
//
//			}
//
//			if(ifVisualize){
//
//				visualizeOptimizationHistory();
//			}
//
//			break;
//		}
//
//	} /* end of the optimization loop */
//
//}
//





void Optimizer::cleanDoEFiles(void) const{

	std::string fileNameObjectiveFunction = objFun.getName()+".csv";
	if(file_exist(fileNameObjectiveFunction)){

		remove(fileNameObjectiveFunction.c_str());
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::string fileNameConstraint = it->getName()+".csv";
		if(file_exist(fileNameConstraint)){

			remove(fileNameConstraint.c_str());
		}


	}


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



void Optimizer::performDoE(unsigned int howManySamples, DoE_METHOD methodID){

	if(ifDisplay){

		std::cout<<"performing DoE...\n";
	}



	if(!ifBoxConstraintsSet){

		cout<<"ERROR: Cannot run DoE before the box-constraints are set!\n";
		abort();
	}



	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
		isHistoryFileInitialized = true;

	}



	mat sampleCoordinates;


	if(methodID == LHS){

		LHSSamples DoE(dimension, lowerBounds, upperBounds, howManySamples);

		if(numberOfDisceteVariables>0){

			DoE.setDiscreteParameterIndices(indicesForDiscreteVariables);
			DoE.setDiscreteParameterIncrements(incrementsForDiscreteVariables);
			DoE.roundSamplesToDiscreteValues();

		}


		std::string filename= this->name + "_samples.csv";
		DoE.saveSamplesToCSVFile(filename);
		sampleCoordinates = DoE.getSamples();
	}
	else{

		cout<<"ERROR: Cannot run DoE with any option other than LHS!\n";
		abort();

	}


#if 0
	printMatrix(sampleCoordinates,"sampleCoordinates");
#endif


	for(unsigned int sampleID=0; sampleID<howManySamples; sampleID++){

		if(ifDisplay){

			std::cout<<"\n##########################################\n";
			std::cout<<"Evaluating sample "<<sampleID<<"\n";

		}

		rowvec dv = sampleCoordinates.row(sampleID);
		Design currentDesign(dv);
		currentDesign.setNumberOfConstraints(numberOfConstraints);
		currentDesign.saveDesignVector(designVectorFileName);


		std::string filenameCVS = objFun.getName()+".csv";

		if(ifDisplay){

			std::cout<<"Appending to data: "<<filenameCVS<<"\n";

		}

		if(!objFun.checkIfGradientAvailable()) {

			//CEO			objFun.evaluate(currentDesign);
			//CEO			objFun.readEvaluateOutput(currentDesign);
			rowvec temp(dimension+1);
			copyRowVector(temp,dv);
			temp(dimension) = currentDesign.trueValue;
			appendRowVectorToCSVData(temp,filenameCVS);


		}
		else{

			//CEO			objFun.evaluateAdjoint(currentDesign);
			//CEO			objFun.readEvaluateOutput(currentDesign);
			rowvec temp(2*dimension+1);
			copyRowVector(temp,currentDesign.designParameters);
			temp(dimension) = currentDesign.trueValue;
			copyRowVector(temp,currentDesign.gradient, dimension+1);


			appendRowVectorToCSVData(temp,filenameCVS);


		}

		if(ifConstrained()){
			computeConstraintsandPenaltyTerm(currentDesign);
		}
		else{
			currentDesign.isDesignFeasible = true;
		}


		calculateImprovementValue(currentDesign);

		addConstraintValuesToDoEData(currentDesign);

		updateOptimizationHistory(currentDesign);


	} /* end of sample loop */




}

void Optimizer::displayMessage(std::string inputString) const{


	if(ifDisplay){

		std::cout<<inputString<<"\n";


	}


}



