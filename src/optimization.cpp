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


COptimizer::COptimizer(std::string nameTestcase, int numberOfOptimizationParams, std::string problemType){

	/* RoDeO does not allow problems with too many optimization parameters */

	if(numberOfOptimizationParams > 100){

		std::cout<<"Problem dimension of the optimization is too large!"<<std::endl;
		abort();

	}

	name = nameTestcase;
	dimension = numberOfOptimizationParams;
	sampleDim = dimension;
	numberOfConstraints = 0;

	maxNumberOfSamples  = 0;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);

	ifBoxConstraintsSet = false;
	iterMaxEILoop = dimension*10000;
	iterGradientEILoop = 100;
	optimizationType = problemType;
	ifVisualize = false;
	howOftenTrainModels = 10; /* train surrogates in every 10 iteration */
	ifCleanDoeFiles = false;
	designVectorFileName = "None";

}

bool COptimizer::checkSettings(void) const{

	bool ifAllSettingsOk = true;

	if(!ifBoxConstraintsSet){

		ifAllSettingsOk = false;

	}

	return ifAllSettingsOk;
}


void COptimizer::setProblemType(std::string type){

	if(type == "MAXIMIZATION" || type == "Maximize" || type == "maximization" || type == "maximize" ){

		type = "maximize";

	}

	if(type == "MINIMIZATION" || type == "Minimize" || type == "minimization" || type == "minimization"){

		type = "minimize";

	}


	optimizationType = type;



}

void COptimizer::setInitialObjectiveFunctionValue(double value){

	initialobjectiveFunctionValue = value;
	IfinitialValueForObjFunIsSet = true;

}

void COptimizer::setMaximumNumberOfIterations(unsigned int maxIterations){

	maxNumberOfSamples = maxIterations;

}

void COptimizer::setFileNameDesignVector(std::string filename){

	designVectorFileName = filename;

}


void COptimizer::setBoxConstraints(std::string filename){

	std::cout<<"Setting box constraints for "<<name<<std::endl;

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

void COptimizer::setBoxConstraints(double lowerBound, double upperBound){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	assert(lowerBound < upperBound);
	lowerBounds.fill(lowerBound);
	upperBounds.fill(upperBound);
	ifBoxConstraintsSet = true;

}


void COptimizer::setBoxConstraints(vec lb, vec ub){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	for(unsigned int i=0; i<dimension; i++) assert(lb(i) < ub(i));

	lowerBounds = lb;
	upperBounds = ub;
	ifBoxConstraintsSet = true;

}


void COptimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;
	sampleDim++;

}


void COptimizer::addObjectFunction(ObjectiveFunction &objFunc){

	assert(ifObjectFunctionIsSpecied == false);
	objFun = objFunc;
	sampleDim++;
	ifObjectFunctionIsSpecied = true;

}





void COptimizer::evaluateConstraints(Design &d){


	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){


		if(!it->checkIfGradientAvailable()){

			it->evaluate(d);

		}else{

			it->evaluateAdjoint(d);

		}
		it->readEvaluateOutput(d);


	}


}

void COptimizer::addConstraintValuesToDoEData(Design &d) const{

	unsigned int countConstraintWithGradient = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){


		if(!it->checkIfGradientAvailable()){

			rowvec saveBuffer(dimension+1);
			copyRowVector(saveBuffer,d.designParameters);
			saveBuffer(dimension) = d.constraintTrueValues(it->getID()-1);
			appendRowVectorToCSVData(saveBuffer,it->getName()+".csv");


		}else{

			rowvec saveBuffer(2*dimension+1);
			copyRowVector(saveBuffer,d.designParameters);
			saveBuffer(dimension) = d.constraintTrueValues(it->getID()-1);

			rowvec gradient = d.constraintGradients[countConstraintWithGradient];
			countConstraintWithGradient++;
			copyRowVector(saveBuffer,gradient,dimension+1);

			appendRowVectorToCSVData(saveBuffer,it->getName()+".csv");

		}


	}


}




void COptimizer::estimateConstraints(CDesignExpectedImprovement &design) const{

	rowvec x = design.dv;
	assert(design.constraintValues.size() == numberOfConstraints);

	unsigned int constraintIt = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		design.constraintValues(constraintIt) = it->interpolate(x);
		constraintIt++;
	}


}




bool COptimizer::checkBoxConstraints(void) const{

	bool flagWithinBounds = true;

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) flagWithinBounds = false;
	}

	return flagWithinBounds;
}



bool COptimizer::checkConstraintFeasibility(rowvec constraintValues) const{

	bool flagFeasibility = true;
	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		flagFeasibility = it->checkFeasibility(constraintValues(i));

		if(flagFeasibility == false) {

			break;
		}
		i++;
	}

	return flagFeasibility;
}



void COptimizer::print(void) const{

	printf("....... %s optimization using max %d samples .........\n",name.c_str(),maxNumberOfSamples);
	printf("Problem dimension = %d\n",dimension);

	objFun.print();

	printConstraints();

	if (constraintFunctions.begin() == constraintFunctions.end()){

		std::cout << "Optimization problem does not have any constraints\n";
	}


}

void COptimizer::printConstraints(void) const{

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->print();

	}



}

void COptimizer::visualizeOptimizationHistory(void) const{

	if(dimension == 2){

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_2d_opthist.py "+ name;
#if 0
		cout<<python_command<<"\n";
#endif
		FILE* in = popen(python_command.c_str(), "r");

		fprintf(in, "\n");

	}


}


void COptimizer::initializeSurrogates(void){

	objFun.initializeSurrogate();

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->initializeSurrogate();

	}

	ifSurrogatesAreInitialized = true;
}


void COptimizer::trainSurrogates(void){

	printf("Training surrogate model for the objective function...\n");
	objFun.trainSurrogate();

	if(constraintFunctions.size() !=0){

		printf("Training surrogate model for the constraints...\n");
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->trainSurrogate();

	}




}



void COptimizer::updateOptimizationHistory(Design d) {

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



void COptimizer::addPenaltyToExpectedImprovementForConstraints(CDesignExpectedImprovement &designCalculated) const{

	if(numberOfConstraints > 0){

		estimateConstraints(designCalculated);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(designCalculated.constraintValues);

		if(!ifConstraintsSatisfied){

			designCalculated.valueExpectedImprovement = 0.0;

		}

	}


}


void COptimizer::computeConstraintsandPenaltyTerm(Design &d) {


	if(constraintFunctions.size() > 0){


		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);

		double penaltyTerm = 0.0;

		if(!ifConstraintsSatisfied){

			std::cout<<"The new sample does not satisfy all the constraints\n";


			if(optimizationType == "minimize"){

				penaltyTerm = LARGE;
				d.isDesignFeasible = false;
			}
			else if(optimizationType == "maximize"){

				penaltyTerm = -LARGE;
				d.isDesignFeasible = false;
			}
			else{

				abort();
			}


		}


		d.objectiveFunctionValue = d.trueValue + penaltyTerm;

	}

}

void COptimizer::addConstraintValuesToData(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->addDesignToData(d);

	}


}

void COptimizer::checkIfSettingsAreOK(void) const{

	if (maxNumberOfSamples == 0){

		fprintf(stderr, "ERROR: Maximum number of samples is not set for the optimization!\n");
		cout<<"maxNumberOfSamples = "<<maxNumberOfSamples<<"\n";
		abort();
	}



	if(checkBoxConstraints() == false){

		fprintf(stderr, "ERROR: Box constraints are not set properly!\n");
		abort();

	}




}


Design COptimizer::findTheGlobalOptimalDesign(void){


	uword indexMin = index_max(this->optimizationHistory.col(this->sampleDim));

	rowvec bestSample = optimizationHistory.row(indexMin);

	Design bestDesign(dimension);

	rowvec dv(dimension);
	for(unsigned int i=0; i<dimension; i++){

		dv(i) = bestSample(i);
	}

	bestDesign.designParameters  = dv;
	bestDesign.trueValue = bestSample(dimension);

	rowvec constraintValues(numberOfConstraints);
	for(unsigned int i=0; i<numberOfConstraints; i++){

		constraintValues(i) = bestSample(i+dimension+1);
	}

	bestDesign.constraintTrueValues = constraintValues;


	return bestDesign;


}



/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void COptimizer::findTheMostPromisingDesign(unsigned int howManyDesigns){

	assert(ifSurrogatesAreInitialized);

	theMostPromisingDesigns.clear();


	CDesignExpectedImprovement designWithMaxEI(dimension,numberOfConstraints);

#pragma omp parallel for
	for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){


		CDesignExpectedImprovement designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVector();

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToExpectedImprovementForConstraints(designToBeTried);

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueExpectedImprovement > designWithMaxEI.valueExpectedImprovement){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found\n");
			designToBeTried.print();
#endif
		}

	}


#pragma omp parallel for
	for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){


		CDesignExpectedImprovement designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVectorAroundASample(designWithMaxEI.dv);

		objFun.calculateExpectedImprovement(designToBeTried);
		addPenaltyToExpectedImprovementForConstraints(designToBeTried);

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueExpectedImprovement > designWithMaxEI.valueExpectedImprovement){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found (second loop) \n");
			designToBeTried.print();
#endif
		}

	}


	theMostPromisingDesigns.push_back(designWithMaxEI);

}


CDesignExpectedImprovement COptimizer::getDesignWithMaxExpectedImprovement(void) const{

	return this->theMostPromisingDesigns.front();


}

/* calculate the gradient of the Expected Improvement function
 * w.r.t design variables by finite difference approximations */
rowvec COptimizer::calculateEIGradient(CDesignExpectedImprovement &currentDesign) const{


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

		double EIplus = currentDesign.valueExpectedImprovement;
		currentDesign.dv(i) -= 2*epsilon;

		objFun.calculateExpectedImprovement(currentDesign);

		double EIminus = currentDesign.valueExpectedImprovement;;


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



CDesignExpectedImprovement COptimizer::MaximizeEIGradientBased(CDesignExpectedImprovement initialDesign) const {

	rowvec gradEI(dimension);
	double stepSize0 = 0.001;
	double stepSize = 0.0;



	objFun.calculateExpectedImprovement(initialDesign);
	addPenaltyToExpectedImprovementForConstraints(initialDesign);


	double EI0 = initialDesign.valueExpectedImprovement;
	CDesignExpectedImprovement bestDesign = initialDesign;

	bool breakOptimization = false;

	for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){

#if 0
		printf("\nGradient search iteration = %d\n", iterGradientSearch);
#endif

		gradEI = calculateEIGradient(bestDesign);

		/* save the design vector */
		CDesignExpectedImprovement dvLineSearchSave = bestDesign ;

#if 0
		printf("Line search...\n");
#endif

		stepSize = stepSize0;

		while(1){


			/* design update */

			bestDesign.gradientUpdateDesignVector(gradEI,stepSize);

			objFun.calculateExpectedImprovement(bestDesign);
			addPenaltyToExpectedImprovementForConstraints(bestDesign);


#if 0
			printf("EI_LS = %15.10f\n",bestDesign.valueExpectedImprovement );

#endif

			/* if ascent is achieved */
			if(bestDesign.valueExpectedImprovement > EI0){
#if 0
				printf("Ascent is achieved with difference = %15.10f\n", bestDesign.valueExpectedImprovement -  EI0);

				bestDesign.print();
#endif
				EI0 = bestDesign.valueExpectedImprovement;
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
void COptimizer::prepareOptimizationHistoryFile(void) const{

	std::string header;
	for(unsigned int i=0; i<dimension; i++){

		header+="x";
		header+=std::to_string(i+1);
		header+=",";

	}

	header+="Objective Function,";


	for(unsigned int i=0; i<this->numberOfConstraints; i++){
		header+="Constraint";
		header+=std::to_string(i+1);

		header+=",";

	}

	header+="Improvement";
	header+="\n";

	std::ofstream optimizationHistoryFile;
	optimizationHistoryFile.open (optimizationHistoryFileName);
	optimizationHistoryFile << header;
	optimizationHistoryFile.close();



}

void COptimizer::clearOptimizationHistoryFile(void) const{

	remove(optimizationHistoryFileName.c_str());

}


void COptimizer::EfficientGlobalOptimization(void){


	checkIfSettingsAreOK();

	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();

	}

	/* main loop for optimization */
	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;


	initializeSurrogates();

	while(1){


		iterOpt++;


#if 0
		printf("Optimization Iteration = %d\n",iterOpt);
#endif


		if(simulationCount%howOftenTrainModels == 0) {

			trainSurrogates();
		}


		findTheMostPromisingDesign();

		CDesignExpectedImprovement optimizedDesignGradientBased = MaximizeEIGradientBased(theMostPromisingDesigns.at(0));


		optimizedDesignGradientBased.print();



		rowvec best_dvNorm = optimizedDesignGradientBased.dv;
		rowvec best_dv =normalizeRowVectorBack(best_dvNorm, lowerBounds, upperBounds);
		double estimatedBestdv = objFun.interpolate(best_dvNorm,true);

#if 1
		printf("The most promising design (not normalized):\n");
		best_dv.print();
		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";

#endif

		Design currentBestDesign(best_dv);
		currentBestDesign.setNumberOfConstraints(numberOfConstraints);
		currentBestDesign.saveDesignVector(designVectorFileName);


		/* now make a simulation for the most promising design */

		if(!objFun.checkIfGradientAvailable()) {

			objFun.evaluate(currentBestDesign);
		}
		else{

			objFun.evaluateAdjoint(currentBestDesign);
		}
		objFun.readEvaluateOutput(currentBestDesign);
		objFun.addDesignToData(currentBestDesign);


		computeConstraintsandPenaltyTerm(currentBestDesign);

		calculateImprovementValue(currentBestDesign);


		if(currentBestDesign.checkIfHasNan()){

			cout<<"ERROR: NaN while reading external executable outputs!\n";
			abort();

		}

		currentBestDesign.print();

		addConstraintValuesToData(currentBestDesign);
		updateOptimizationHistory(currentBestDesign);


		Design globalBestDesign = findTheGlobalOptimalDesign();
		std::cout<<"######## Best Design ############\n\n";
		globalBestDesign.print();
		std::cout<<"\n\n";


		simulationCount ++;

		/* terminate optimization */
		if(simulationCount >= maxNumberOfSamples){

			printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");


			if(ifVisualize){

				visualizeOptimizationHistory();
			}



			break;
		}



	} /* end of the optimization loop */



}
void COptimizer::cleanDoEFiles(void) const{

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

void COptimizer::calculateImprovementValue(Design &d){

	if(d.isDesignFeasible){

		if(!IfinitialValueForObjFunIsSet){

			initialobjectiveFunctionValue  = d.objectiveFunctionValue;
			IfinitialValueForObjFunIsSet = true;

		}


		if(optimizationType == "minimize"){

			if(d.objectiveFunctionValue < initialobjectiveFunctionValue){

				d.improvementValue = initialobjectiveFunctionValue - d.objectiveFunctionValue;

			}

		}
		if(optimizationType == "maximize"){

			if(d.objectiveFunctionValue > initialobjectiveFunctionValue){

				d.improvementValue = d.objectiveFunctionValue - initialobjectiveFunctionValue;

			}

		}


	}


}



void COptimizer::performDoE(unsigned int howManySamples, DoE_METHOD methodID){

	if(!ifBoxConstraintsSet){

		cout<<"ERROR: Cannot run DoE before the box-constraints are set!\n";
		abort();
	}


	if(ifCleanDoeFiles){

		cleanDoEFiles();
	}

	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
		isHistoryFileInitialized = true;

	}



	mat sampleCoordinates;


	if(methodID == LHS){

		LHSSamples DoE(dimension, lowerBounds, upperBounds, howManySamples);

		std::string filename= this->name + "_samples.csv";
		DoE.saveSamplesToCSVFile(filename);
		sampleCoordinates = DoE.getSamples();
	}

#if 0
	printMatrix(sampleCoordinates,"sampleCoordinates");
#endif


	for(unsigned int sampleID=0; sampleID<howManySamples; sampleID++){

		std::cout<<"Evaluating sample "<<sampleID<<"\n";
		std::cout<<"##########################################\n";

		rowvec dv = sampleCoordinates.row(sampleID);
		Design currentDesign(dv);
		currentDesign.setNumberOfConstraints(numberOfConstraints);
		currentDesign.saveDesignVector(designVectorFileName);




		if(!objFun.checkIfGradientAvailable()) {

			objFun.evaluate(currentDesign);
			objFun.readEvaluateOutput(currentDesign);
			rowvec temp(dimension+1);
			copyRowVector(temp,dv);
			temp(dimension) = currentDesign.trueValue;
			appendRowVectorToCSVData(temp,objFun.getName()+".csv");


		}
		else{

			objFun.evaluateAdjoint(currentDesign);
			objFun.readEvaluateOutput(currentDesign);
			rowvec temp(2*dimension+1);
			copyRowVector(temp,currentDesign.designParameters);
			temp(dimension) = currentDesign.trueValue;
			copyRowVector(temp,currentDesign.gradient, dimension+1);


			appendRowVectorToCSVData(temp,objFun.getName()+".csv");


		}

		computeConstraintsandPenaltyTerm(currentDesign);

		calculateImprovementValue(currentDesign);

		currentDesign.print();

		addConstraintValuesToDoEData(currentDesign);

		updateOptimizationHistory(currentDesign);



	} /* end of sample loop */




}




