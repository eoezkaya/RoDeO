/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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







COptimizer::COptimizer(std::string nameTestcase, int numberOfOptimizationParams, std::string problemType = "minimize"){

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
	//	dataMin.zeros(dimension);
	//	dataMax.zeros(dimension);
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



void COptimizer::estimateConstraints(rowvec x, rowvec &constraintValues) const{

	unsigned int contraintIt = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		constraintValues(contraintIt) = it->interpolate(x);
		contraintIt++;
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

	rowvec newSample(sampleDim);

	for(unsigned int i=0; i<dimension; i++) {

		newSample(i) = d.designParameters(i);

	}

	newSample(dimension) = d.trueValue;


	for(unsigned int i=0; i<numberOfConstraints; i++){

		newSample(i+dimension+1) = 	d.constraintTrueValues(i);
	}



	optimizationHistory.insert_rows( optimizationHistory.n_rows, newSample );
	appendRowVectorToCSVData(newSample,"optimizationHistory.csv");

#if 0
	printf("optimizationHistory:\n");
	optimizationHistory.print();

#endif


}

double COptimizer::computeEIPenaltyForConstraints(rowvec dv) const{

	double penaltyTerm = 0.0;

	if(constraintFunctions.size() > 0){
		rowvec estimatesForConstaints(constraintFunctions.size());

		estimateConstraints(dv,estimatesForConstaints);
#if 0
		if(tid ==0)
		{
			printVector(estimatesForConstaints,"estimatesForConstaints");

		}
#endif
		bool ifConstraintsSatisfied = checkConstraintFeasibility(estimatesForConstaints);

		if(!ifConstraintsSatisfied){


			penaltyTerm = -LARGE;

		}

	}


	return penaltyTerm;
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
			}
			else if(optimizationType == "maximize"){

				penaltyTerm = -LARGE;
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


//rowvec COptimizer::generateRandomRowVectorAroundASample(void){
//
//	unsigned int randomIndx = generateRandomInt(0,maxNumberOfSamples-1);
//
//
//
//}

/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */
void COptimizer::findTheMostPromisingDesign(unsigned int howManyDesigns){


	theMostPromisingDesigns.clear();

	double maxEI = 0.0;
	rowvec bestDesignVector(dimension);

#pragma omp parallel for
	for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){


		rowvec dvNotNormalized = generateRandomRowVector(lowerBounds, upperBounds);

		rowvec dv = normalizeRowVector(dvNotNormalized,lowerBounds, upperBounds);
#if 0
		printVector(dv);
#endif

		double EI = objFun.calculateExpectedImprovement(dv);

		double penaltyTerm = computeEIPenaltyForConstraints(dv);

		EI = EI + penaltyTerm;

		if(EI > maxEI){

			bestDesignVector = dv;
			maxEI = EI;
#if 0
			printf("A design with a better EI value has been found, EI = %15.10f\n", EI);
			best_dv.print();
#endif
		}

	}

	CDesignExpectedImprovement bestDesign(5,2);
	bestDesign.dv = bestDesignVector;
	bestDesign.valueExpectedImprovement = maxEI;

	theMostPromisingDesigns.push_back(bestDesign);

}


/* calculate the gradient of the Expected Improvement function
 * w.r.t design variables by finite difference approximations */
rowvec COptimizer::calculateEIGradient(rowvec designVector) const{


	rowvec gradient(dimension);

	for(unsigned int i=0; i<dimension; i++){
#if 0
		printf("dv:\n");
		dvGradientSearch.print();
#endif

		double dvSave = designVector(i);


#if 0
		printf("epsilon_EI = %15.10f\n",epsilon_EI);
#endif

		double epsilon = designVector(i)*0.00001;
		designVector(i) += epsilon;

#if 0
		printf("dv perturbed:\n");
		dvPerturbed.print();
#endif


		double EIplus = objFun.calculateExpectedImprovement(designVector);
		designVector(i) -= 2*epsilon;
		double EIminus = objFun.calculateExpectedImprovement(designVector);


		/* obtain the forward finite difference quotient */
		double fdVal = (EIplus - EIminus)/(2*epsilon);
		gradient(i) = fdVal;
		designVector(i) = dvSave;


	} /* end of finite difference loop */
#if 0
	printf("Gradient vector:\n");
	gradEI.print();
#endif

	return gradient;
}

rowvec COptimizer::designVectorGradientUpdate(rowvec dv0, rowvec gradient, double stepSize) const{

	rowvec dvUpdated(dimension);
	for(unsigned int k=0; k<dimension; k++){

		dvUpdated(k) = dv0(k) + stepSize*gradient(k);

	}

	rowvec dvUpdatedNotNormalized = normalizeRowVectorBack(dvUpdated, lowerBounds, upperBounds);



	for(unsigned int k=0; k<dimension; k++){

		/* if new design vector does not satisfy the box constraints in original coordinates */
		if(dvUpdatedNotNormalized(k) < lowerBounds(k)) dvUpdatedNotNormalized(k) = lowerBounds(k);
		if(dvUpdatedNotNormalized(k) > upperBounds(k)) dvUpdatedNotNormalized(k) = upperBounds(k);

	}

	/* return the design vector in normalized coordinates */
	dvUpdated = normalizeRowVector(dvUpdatedNotNormalized,lowerBounds,upperBounds);

	return dvUpdated;


}

CDesignExpectedImprovement COptimizer::MaximizeEIGradientBased(CDesignExpectedImprovement initialDesignVector) const {

	rowvec gradEI(dimension);
	double stepSize0 = 0.0001;
	double stepSize = 0.0;
	rowvec bestDesignVector = initialDesignVector.dv;
	double EI0 = initialDesignVector.valueExpectedImprovement;

	bool breakOptimization = false;

	for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){

#if 1
		printf("\nGradient search iteration = %d\n", iterGradientSearch);
#endif

		gradEI = calculateEIGradient(bestDesignVector);



		/* save the design vector */
		rowvec dvLineSearchSave = bestDesignVector ;

#if 0
		printf("Line search...\n");
#endif

		stepSize = stepSize0;

		while(1){


			/* design update */


			bestDesignVector = designVectorGradientUpdate(bestDesignVector,gradEI, stepSize);



			double EI_LS = objFun.calculateExpectedImprovement(bestDesignVector);
			double penaltyTerm = computeEIPenaltyForConstraints(bestDesignVector);

			EI_LS = EI_LS + penaltyTerm;

#if 1
			printf("EI_LS = %15.10f\n",EI_LS);

#endif

			/* if ascent is achieved */
			if(EI_LS > EI0){
#if 1
				printf("Ascent is achieved with difference = %15.10f\n", EI_LS- EI0);
#endif
				EI0 = EI_LS;
				break;
			}

			else{ /* else halve the stepsize and set design to initial */

				stepSize = stepSize * 0.5;
				bestDesignVector = dvLineSearchSave;
#if 1
				printf("stepsize = %15.10f\n",stepSize);

#endif
				if(stepSize < 10E-12) {
#if 1
					printf("The stepsize is getting too small!\n");
#endif

					breakOptimization = true;
					break;
				}
			}

		}
#if 1


#endif

		if(breakOptimization) break;

	} /* end of gradient-search loop */


	CDesignExpectedImprovement result(3,3);
//	result.dv = bestDesignVector;
//	result.valueExpectedImprovement = EI0;

	return result;


}
void COptimizer::prepareOptimizationHistoryFile(void) const{

	std::string header;
	for(unsigned int i=0; i<dimension; i++){

		header+="x";
		header+=std::to_string(i+1);
		header+=",";

	}

	if(this->numberOfConstraints == 0){
		header+="Objective Function";

	}
	else{
		header+="Objective Function,";

	}



	for(unsigned int i=0; i<this->numberOfConstraints; i++){
		header+="Constraint";
		header+=std::to_string(i+1);

		if(i < this->numberOfConstraints -1){

			header+=",";
		}

	}
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

	clearOptimizationHistoryFile();
	prepareOptimizationHistoryFile();

	/* main loop for optimization */
	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;

	double bestObjFunVal = LARGE;
	rowvec best_dvGlobal(dimension);
	unsigned int bestIndx = -1;

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


		if(currentBestDesign.checkIfHasNan()){

			cout<<"ERROR: NaN while reading external executable outputs!\n";
			abort();



		}



		addConstraintValuesToData(currentBestDesign);
		updateOptimizationHistory(currentBestDesign);
		if(currentBestDesign.objectiveFunctionValue < bestObjFunVal){

			bestIndx = iterOpt;
			bestObjFunVal = currentBestDesign.objectiveFunctionValue;
			best_dvGlobal = currentBestDesign.designParameters;
#if 0
			printf("\nBetter design has been found:\n");
			printf("dv =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",objFunVal);
#endif

		}




		simulationCount ++;

		/* terminate optimization */
		if(simulationCount >= maxNumberOfSamples){

			printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");
			printf("Global optimal solution:\n");
			printf("design vector =");
			best_dvGlobal.print();
			printf("Objective function value = %15.10f\n",bestObjFunVal);
			printf("Index = %d\n",bestIndx);

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





void COptimizer::performDoE(unsigned int howManySamples, DoE_METHOD methodID){

	if(!ifBoxConstraintsSet){

		cout<<"ERROR: Cannot run DoE before the box-constraints are set!\n";
		abort();
	}


	if(ifCleanDoeFiles){

		cleanDoEFiles();
	}


	mat sampleCoordinates;


	if(methodID == LHS){

		LHSSamples DoE(dimension, lowerBounds, upperBounds, howManySamples);


		//		if(dimension == 2) DoE.visualize();
		std::string filename= this->name + "_samples.csv";
		DoE.saveSamplesToCSVFile(filename);
		sampleCoordinates = DoE.getSamples();
	}

#if 1
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

		evaluateConstraints(currentDesign);

		addConstraintValuesToDoEData(currentDesign);



	} /* end of sample loop */




}
//void testOptimizationWingweight(void){
//	/*
//	Sw: Wing Area (ft^2) (150,200)
//	 *  Wfw: Weight of fuel in the wing (lb) (220,300)
//	 *  A: Aspect ratio (6,10)
//	 *  Lambda: quarter chord sweep (deg) (-10,10)
//	 *  q: dynamic pressure at cruise (lb/ft^2)  (16,45)
//	 *  lambda: taper ratio (0.5,1)
//	 *  tc: aerofoil thickness to chord ratio (0.08,0.18)
//	 *  Nz: ultimate load factor (2.5,6)
//	 *  Wdg: flight design gross weight (lb)  (1700,2500)
//	 *  Wp: paint weight (lb/ft^2) (0.025, 0.08)
//
//
//	 */
//	vec lb(10);
//	vec ub(10);
//	lb(0) = 150; ub(0) = 200;
//	lb(1) = 220; ub(1) = 300;
//	lb(2) = 6;   ub(2) = 10;
//	lb(3) = -10; ub(3) = 10;
//	lb(4) = 16; ub(4) = 45;
//	lb(5) = 0.5; ub(5) = 1;
//	lb(6) = 0.08; ub(6) = 0.18;
//	lb(7) = 2.5; ub(7) = 6;
//	lb(8) = 1700; ub(8) = 2500;
//	lb(9) = 0.025; ub(9) = 0.08;
//
//
//
//	std::string problemName = "Wingweight";
//	unsigned int dimension = 10;
//	Optimizer OptimizationStudy(problemName, dimension);
//
//	ObjectiveFunction objFunc(problemName, Wingweight, dimension);
//	OptimizationStudy.addObjectFunction(objFunc);
//
//	OptimizationStudy.maxNumberOfSamples = 100;
//
//	OptimizationStudy.setBoxConstraints(lb,ub);
//	OptimizationStudy.performDoE(100,LHS);
//
//
//	OptimizationStudy.EfficientGlobalOptimization();
//
//
//}
//
//void testOptimizationEggholder(void){
//
//
//	std::string problemName = "Eggholder";
//	unsigned int dimension = 2;
//	Optimizer OptimizationStudy(problemName, dimension);
//
//	ObjectiveFunction objFunc(problemName, Eggholder, dimension);
//	OptimizationStudy.addObjectFunction(objFunc);
//
//	OptimizationStudy.maxNumberOfSamples = 50;
//
//	OptimizationStudy.setBoxConstraints(0.0,200.0);
//	OptimizationStudy.performDoE(50,LHS);
//
//	OptimizationStudy.ifVisualize = true;
//	OptimizationStudy.EfficientGlobalOptimization();
//
//
//}
//
//void testOptimizationHimmelblau(void){
//
//
//	std::string problemName = "Himmelblau";
//	unsigned int dimension = 2;
//	Optimizer OptimizationStudy(problemName, dimension);
//
//	ObjectiveFunction objFunc(problemName, Himmelblau, dimension);
//	OptimizationStudy.addObjectFunction(objFunc);
//
//	OptimizationStudy.maxNumberOfSamples = 100;
//
//	OptimizationStudy.setBoxConstraints(-5.0,5.0);
//	OptimizationStudy.performDoE(50,LHS);
//
//	OptimizationStudy.ifVisualize = true;
//	OptimizationStudy.EfficientGlobalOptimization();
//
//
//}
//
//void testOptimizationHimmelblauExternalExe(void){
//
//
//	std::string problemName = "Himmelblau";
//	unsigned int dimension = 2;
//	Optimizer OptimizationStudy(problemName, dimension);
//
//	ObjectiveFunction objFunc(problemName, dimension);
//	objFunc.setFileNameReadObjectFunction("objFunVal.dat");
//	objFunc.setExecutablePath("/home/emre/RoDeO/Tests/HimmelblauOptimization");
//	objFunc.setExecutableName("himmelblau");
//	objFunc.setFileNameDesignVector("dv.csv");
//	OptimizationStudy.addObjectFunction(objFunc);
//
//	OptimizationStudy.maxNumberOfSamples = 100;
//
//	OptimizationStudy.setBoxConstraints(-5.0,5.0);
//	OptimizationStudy.performDoE(50,LHS);
//
//	OptimizationStudy.ifVisualize = true;
//	OptimizationStudy.EfficientGlobalOptimization();
//
//
//}




