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
#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, double (*objFun)(double *), unsigned int dimension)
: surrogateModel(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = objFun;

	assert(dim < 1000);


}

ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, unsigned int dimension)
: surrogateModel(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;
	objectiveFunPtr = empty;
	executableName = "None";
	executablePath = "None";
	fileNameObjectiveFunctionRead = "None";
	fileNameDesignVector = "None";
	assert(dim < 1000);


}


ObjectiveFunction::ObjectiveFunction(){

	dim = 0;
	name = "None";
	objectiveFunPtr = empty;
	executableName = "None";
	executablePath = "None";
	fileNameObjectiveFunctionRead = "None";
	fileNameDesignVector = "None";


}

void ObjectiveFunction::readConfigFile(void){

	cout<<"Reading configuration file:"<<settings.config_file<<"\n";

	if(!file_exist(settings.config_file.c_str())){

		cout<<"ERROR: Configuration file "<<settings.config_file<<" does not exist!\n";
		abort();
	}


}



void ObjectiveFunction::setFileNameReadObjectFunction(std::string fileName){

	fileNameObjectiveFunctionRead = fileName;

}


void ObjectiveFunction::setFileNameDesignVector(std::string fileName){

	fileNameDesignVector = fileName;

}

void ObjectiveFunction::setExecutablePath(std::string path){

	executablePath = path;

}

void ObjectiveFunction::setExecutableName(std::string exeName){

	executableName = exeName;

}

void ObjectiveFunction::trainSurrogate(void){

	surrogateModel.initializeSurrogateModel();
	surrogateModel.train();

}



void ObjectiveFunction::saveDoEData(mat data) const{

	std::string fileName = surrogateModel.getInputFileName();
	data.save(fileName,csv_ascii);


}

double ObjectiveFunction::calculateExpectedImprovement(rowvec x){

	return surrogateModel.calculateExpectedImprovement(x);

}


double ObjectiveFunction::evaluate(rowvec x,bool ifAddToData = true){


	double functionValue = 0.0;

	if( objectiveFunPtr != empty){

		functionValue =  objectiveFunPtr(x.memptr());

	}

	else if (fileNameObjectiveFunctionRead != "None" && executableName != "None" && fileNameDesignVector != "None"){


		bool ifSaveIsSuccessful= x.save(fileNameDesignVector,raw_ascii);

		if(!ifSaveIsSuccessful){

			std::cout << "ERROR: There was a problem while saving the design vector!\n";
			abort();

		}

		std::string runCommand;
		if(this->executablePath!= "None") {

			runCommand = executablePath +"/" + executableName;
		}
		else{
			runCommand = "./" + executableName;
		}

		system(runCommand.c_str());

		std::ifstream ifile(fileNameObjectiveFunctionRead, std::ios::in);

		    if (!ifile.is_open()) {

		        std::cout << "ERROR: There was a problem opening the input file!\n";
		        abort();
		    }

		    ifile >> functionValue;

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}

	if(std::isnan(functionValue)){

		cout<<"ERROR: NaN as the objective function value!\n";
		abort();

	}


	if(ifAddToData){

		rowvec newsample(dim +1);

		for(unsigned int k=0; k<dim; k++){

			newsample(k) = x(k);

		}
		newsample(dim) = functionValue;


#if 1
		printf("new sample: \n");
		newsample.print();
#endif

		if(surrogateModel.addNewSampleToData(newsample) !=0){

			printf("Warning: The new sample cannot be added into the training data since it is too close to a sample!\n");

		}


	}

	return functionValue;


}


double ObjectiveFunction::ftilde(rowvec x) const{

	return surrogateModel.interpolate(x);

}

void ObjectiveFunction::print(void) const{

	std::cout<<std::endl;
	std::cout<<"Objective Function"<<std::endl;
	std::cout<<"Name: "<<name<<std::endl;
	std::cout<<"Dimension: "<<dim<<std::endl;
	std::cout<<std::endl;

	surrogateModel.printSurrogateModel();

}


ConstraintFunction::ConstraintFunction(){

	dim = 0;
	name = "None";
	pConstFun = empty;
	inequalityType = "None";
	targetValue = 0.0;

}


ConstraintFunction::ConstraintFunction(std::string constraintName, std::string constraintType, double constraintValue, double (*fun_ptr)(double *), unsigned int dimension, bool ifSurrogate){

	dim = dimension;
	name = constraintName;
	pConstFun = fun_ptr;
	inequalityType = constraintType;

	assert(dim < 1000);
	assert(constraintType == "lt" || constraintType == "gt");


	targetValue = constraintValue;
	ifNeedsSurrogate = ifSurrogate;

	if(ifNeedsSurrogate){

		KrigingModel temp(constraintName);
		surrogateModel = temp;

	}


}


void ConstraintFunction::saveDoEData(mat data) const{

	std::string fileName = surrogateModel.getInputFileName();
	data.save(fileName,csv_ascii);


}


void ConstraintFunction::trainSurrogate(void){

	assert(ifNeedsSurrogate);
	surrogateModel.train();

}


double ConstraintFunction::ftilde(rowvec x) const{

	assert(ifNeedsSurrogate);
	return surrogateModel.interpolate(x);

}

bool ConstraintFunction::checkFeasibility(double value){

	bool result = true;
	if (inequalityType == "lt"){

		if(value > targetValue ){

			result = false;

		}

	}

	if (inequalityType == "gt"){

		if(value < targetValue ){

			result = false;

		}

	}

	return result;
}


double ConstraintFunction::evaluate(rowvec x, bool ifAddToData = true){

	double functionValue =  pConstFun(x.memptr());

	assert(std::isnan(functionValue) == false);


	if(ifAddToData){

		rowvec newsample(dim +1);

		for(unsigned int k=0; k<dim; k++){

			newsample(k) = x(k);

		}
		newsample(dim) = functionValue;


#if 1
		printf("new sample: \n");
		newsample.print();
#endif


		surrogateModel.addNewSampleToData(newsample);

	}

	return functionValue;
}




void ConstraintFunction::print(void) const{

	std::cout<<std::endl;
	std::cout<<"Constraint Function"<<std::endl;
	std::cout<<"Name: "<<name<<std::endl;
	std::cout<<"Dimension: "<<dim<<std::endl;
	std::cout<<"Type of constraint: "<<inequalityType<<" "<<targetValue<<std::endl;
	std::cout<<"Needs surrogate:"<<ifNeedsSurrogate<<std::endl;
	std::cout<<std::endl;

	surrogateModel.printSurrogateModel();

}





OptimizerWithGradients::OptimizerWithGradients(){

	name = "None";
	size_of_dv = 0;
	max_number_of_samples  = 0;
	iterMaxEILoop = 0;
	doesValidationFileExist = false;

}



OptimizerWithGradients::OptimizerWithGradients(int input_size){

	name = "None";
	size_of_dv = input_size;
	max_number_of_samples  = 0;
	lower_bound_dv.zeros(input_size);
	upper_bound_dv.zeros(input_size);
	iterMaxEILoop = input_size*10000;
	doesValidationFileExist = false;

}

OptimizerWithGradients::OptimizerWithGradients(std::string nameTestcase,int input_size){

	name = nameTestcase;
	size_of_dv = input_size;
	max_number_of_samples  = 0;
	lower_bound_dv.zeros(input_size);
	upper_bound_dv.zeros(input_size);
	iterMaxEILoop = input_size*10000;
	doesValidationFileExist = false;

}



void OptimizerWithGradients::print(void){

	printf("....... %s optimization using max %d samples .........\n",name.c_str(),max_number_of_samples);
	if(lower_bound_dv(0) == 0 && upper_bound_dv(0) == 0 ){

		fprintf(stderr, "Error: Box constraints are not set! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);


	}

}

//void OptimizerWithGradients::EfficientGlobalOptimization(void){
//
//	print();
//
//	AggregationModel ObjFunModel(name,size_of_dv);
//
//	if(doesValidationFileExist){
//
//		std::string validation_filename = name + "_validation.csv";
//		ObjFunModel.validationset_input_filename = validation_filename;
//		ObjFunModel.visualizeAggModelValidation = true;
//		ObjFunModel.visualizeKrigingValidation = true;
//		ObjFunModel.visualizeKernelRegressionValidation = false;
//	}
//
//	ObjFunModel.number_of_cv_iterations = 0;
//
//	ObjFunModel.train();
//
//	/* main loop for optimization */
//	unsigned int simulationCount = 0;
//	unsigned int iterOpt=0;
//
//	double bestObjFunVal = LARGE;
//	rowvec best_dvGlobal(size_of_dv);
//	unsigned int bestIndx = -1;
//
//	while(1){
//		iterOpt++;
//#if 1
//		printf("Optimization Iteration = %d\n",iterOpt);
//		printf("Sample minimum = %10.7f\n",ObjFunModel.ymin );
//#endif
//
//		double maxEI = 0.0;
//		rowvec best_dv(size_of_dv);
//
//#pragma omp parallel for
//		for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){
//			rowvec dv(size_of_dv);
//			rowvec dvNorm(size_of_dv);
//
//			/* Generate a random design vector */
//			for(unsigned int k=0; k<size_of_dv; k++){
//
//				dv(k)= generateRandomDouble(lower_bound_dv(k), upper_bound_dv(k));
//				dvNorm(k) = (1.0/ObjFunModel.dim)*(dv(k) - ObjFunModel.xmin(k)) / (ObjFunModel.xmax(k) - ObjFunModel.xmin(k));
//			}
//
//#if 0
//			printf("dv = \n");
//			dv.print();
//#endif
//			double ftilde = 0.0;
//			double ssqr   = 0.0;
//			ObjFunModel.ftilde_and_ssqr(dvNorm,&ftilde,&ssqr);
//
//
//#if 0
//			printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
//#endif
//
//			double	standart_error = sqrt(ssqr)	;
//
//			double EI = 0.0;
//
//			if(standart_error!=0.0){
//
//				double	EIfac = (ObjFunModel.ymin - ftilde)/standart_error;
//
//				/* calculate the Expected Improvement value */
//				EI = (ObjFunModel.ymin - ftilde)*cdf(EIfac,0.0,1.0)+ standart_error * pdf(EIfac,0.0,1.0);
//			}
//			else{
//
//				EI = 0.0;
//
//			}
//#if 0
//			printf("EI value = %15.10f\n",EI);
//#endif
//			if(EI > maxEI){
//
//				best_dv = dv;
//				maxEI = EI;
//#if 1
//				printf("A design with better EI value has been find, EI = %15.10f\n", EI);
//				best_dv.print();
//#endif
//			}
//
//
//		} /* end of EI loop */
//
//		/* now make a simulation for the most promising design */
//
//		rowvec grad(size_of_dv);
//		double fVal = adj_fun(best_dv.memptr(), grad.memptr());
//		double objFunVal = fVal;
//
//		if(objFunVal < bestObjFunVal){
//
//			bestIndx = iterOpt;
//			bestObjFunVal = objFunVal;
//			best_dvGlobal = best_dv;
//#if 1
//			printf("\nBetter design has been found:\n");
//			printf("dv =");
//			best_dv.print();
//			printf("Objective function value = %15.10f\n",objFunVal);
//#endif
//
//		}
//
//
//		simulationCount ++;
//#if 1
//		printf("Simulation at dv = \n");
//		best_dv.print();
//		printf("True value of the function = %15.10f\n",fVal);
//		printf("grad = \n");
//		grad.print();
//
//
//#endif
//
//		/* new row that will be added to the data */
//
//		rowvec newsample(2*size_of_dv +1);
//
//		for(unsigned int k=0; k<size_of_dv; k++){
//
//			newsample(k) = best_dv(k);
//
//		}
//		newsample(size_of_dv) = fVal;
//
//		for(unsigned int k=size_of_dv+1; k<2*size_of_dv+1; k++){
//
//			newsample(k) = grad(k-size_of_dv-1);
//
//		}
//#if 1
//		printf("new sample: \n");
//		newsample.print();
//#endif
//
//
//
//		ObjFunModel.data.resize(ObjFunModel.N+1,2*ObjFunModel.dim+1);
//
//		ObjFunModel.data.row(ObjFunModel.N) = newsample;
//		ObjFunModel.data.save(ObjFunModel.input_filename,csv_ascii);
//		ObjFunModel.update();
//
//
//		if(simulationCount >= max_number_of_samples){
//
//			printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");
//			printf("Global optimal solution:\n");
//			printf("dv =");
//			best_dv.print();
//			printf("Objective function value = %15.10f\n",bestObjFunVal);
//			printf("Index = %d\n",bestIndx);
//			break;
//		}
//
//
//
//	}
//
//
//
//}



Optimizer::Optimizer(std::string nameTestcase, int numberOfOptimizationParams, std::string problemType = "minimize"){

	/* RoDeO does not allow problems with too many optimization parameters */

	if(numberOfOptimizationParams > 100){

		std::cout<<"Problem dimension of the optimization is too large!"<<std::endl;
		abort();

	}

	name = nameTestcase;
	dimension = numberOfOptimizationParams;
	numberOfConstraints = 0;
	maxNumberOfSamples  = 0;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	ifBoxConstraintsSet = false;
	iterMaxEILoop = dimension*10000;
	iterGradientEILoop = 100;
	epsilon_EI = 10E-4;
	optimizationType = problemType;
	ifVisualize = false;
	howOftenTrainModels = 10; /* train surrogates in every 10 iteration */


	assert(optimizationType == "minimize" || optimizationType == "maximize");

}

void Optimizer::setBoxConstraints(std::string filename){

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

void Optimizer::setBoxConstraints(double lowerBound, double upperBound){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	assert(lowerBound < upperBound);
	lowerBounds.fill(lowerBound);
	upperBounds.fill(upperBound);
	ifBoxConstraintsSet = true;

}


void Optimizer::setBoxConstraints(vec lb, vec ub){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	for(unsigned int i=0; i<dimension; i++) assert(lb(i) < ub(i));

	lowerBounds = lb;
	upperBounds = ub;
	ifBoxConstraintsSet = true;

}


void Optimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;

}


void Optimizer::addObjectFunction(ObjectiveFunction &objFunc){

	objFun = objFunc;

}


void Optimizer::evaluateConstraints(rowvec x, rowvec &constraintValues, bool ifAddToData = true){

	unsigned int contraintIt = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		constraintValues(contraintIt) = it->evaluate(x,ifAddToData);
		contraintIt++;
	}


}


bool Optimizer::checkBoxConstraints(void) const{

	bool flagWithinBounds = true;

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) flagWithinBounds = false;
	}

	return flagWithinBounds;
}



bool Optimizer::checkConstraintFeasibility(rowvec constraintValues){

	bool flagFeasibility = true;
	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		flagFeasibility = it->checkFeasibility(constraintValues(i));
		i++;
	}

	return flagFeasibility;
}



void Optimizer::print(void) const{

	printf("....... %s optimization using max %d samples .........\n",name.c_str(),maxNumberOfSamples);
	printf("Problem dimension = %d\n",dimension);

	objFun.print();

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->print();

	}

	if (constraintFunctions.begin() == constraintFunctions.end()){

		std::cout << "Optimization problem does not have any constraints\n";
	}



	printf("epsilon_EI = %15.10f\n",epsilon_EI );

}

void Optimizer::visualizeOptimizationHistory(void) const{

	if(dimension == 2){

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_2d_opthist.py "+ name;
#if 1
		cout<<python_command<<"\n";
#endif
		FILE* in = popen(python_command.c_str(), "r");

		fprintf(in, "\n");

	}


}


void Optimizer::trainSurrogates(void){
	printf("Training surrogate model for the objective function...\n");
	objFun.trainSurrogate();

	if(constraintFunctions.size() !=0){

		printf("Training surrogate model for the constraints...\n");
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->trainSurrogate();

	}


}



void Optimizer::EfficientGlobalOptimization(void){

	remove("optimizationHistory.csv");

	if (maxNumberOfSamples == 0){

		fprintf(stderr, "ERROR: Maximum number of samples is not set for the optimization!\n");
		cout<<"maxNumberOfSamples = "<<maxNumberOfSamples<<"\n";
		abort();
	}



	if(checkBoxConstraints() == false){

		fprintf(stderr, "ERROR: Box constraints are not set properly!\n");
		abort();

	}


#if 1
	print();
#endif

	trainSurrogates();



	/* main loop for optimization */
	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;

	double bestObjFunVal = LARGE;
	rowvec best_dvGlobal(dimension);
	unsigned int bestIndx = -1;

	while(1){
		iterOpt++;
#if 0
		printf("Optimization Iteration = %d\n",iterOpt);
#endif


		if(simulationCount%howOftenTrainModels == 0) {

			trainSurrogates();
		}


		double maxEI = 0.0;

		rowvec best_dv(dimension);


#pragma omp parallel for
		for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){
#if 0
			printf("iterEI = %d\n",iterEI);
#endif


			/* Generate a random design vector and normalize it*/
			rowvec dv = generateRandomRowVector(0.0, 1.0, dimension)*1.0/dimension;



#if 0
			printf("dv = \n");
			dv.print();
#endif

			double EI = objFun.calculateExpectedImprovement(dv);

#if 0
			printf("EI value = %15.10f\n",EI);
#endif


			if(EI > maxEI){

				best_dv = dv;
				maxEI = EI;
#if 0
				printf("A design with a better EI value has been find, EI = %15.10f\n", EI);
				best_dv.print();
#endif
			}


		} /* end of EI loop */




		rowvec gradEI(dimension);
		/* auxilliary vector used for the gradient search */
		rowvec dvGradientSearch = best_dv;

		double EI0 = maxEI;


		/* optimize further from the best design */

		bool breakOptimization = false;

		for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){
#if 0
			printf("\nGradient search iteration = %d\n", iterGradientSearch);
#endif

			for(unsigned int iterFDLoop=0; iterFDLoop<dimension; iterFDLoop++){
#if 0
				printf("dv:\n");
				dvGradientSearch.print();
#endif


				rowvec dvPerturbed = dvGradientSearch;

#if 0
				printf("epsilon_EI = %15.10f\n",epsilon_EI);
#endif

				dvPerturbed(iterFDLoop) += epsilon_EI;

#if 0
				printf("dv perturbed:\n");
				dvPerturbed.print();
#endif


				double EIplus = objFun.calculateExpectedImprovement(dvPerturbed);
#if 0
				printf("FD for parameter: %d, EIplus = %15.10f, EI0 = %15.10f\n", iterFDLoop, EIplus,EI0);
#endif

				/* obtain the forward finite difference quotient */
				double fdVal = (EIplus - EI0)/epsilon_EI;
				gradEI(iterFDLoop) = fdVal;


			} /* end of finite difference loop */

#if 0
			printf("Gradient vector:\n");
			gradEI.print();
#endif


			double stepsize_EI = 0.0001;
			/* save the design vector */
			rowvec dvGradientSearchSave = dvGradientSearch;

#if 0
			printf("Line search...\n");
#endif



			while(1){


				/* design update */

				for(unsigned int k=0; k<dimension; k++){

					dvGradientSearch(k) = dvGradientSearch(k) + stepsize_EI*gradEI(k);

					/* if new design vector does not satisfy the box constraints in normalized coordinates*/

					if(dvGradientSearch(k) < 0.0) dvGradientSearch(k) = 0.0;
					if(dvGradientSearch(k) > 1.0/dimension) dvGradientSearch(k) = 1.0/dimension;

				}

				double EI_LS = objFun.calculateExpectedImprovement(dvGradientSearch);
#if 0
				printf("EI_LS = %15.10f\n",EI_LS);

#endif

				/* if ascent is achieved */
				if(EI_LS > EI0){
#if 0
					printf("Ascent is achieved with difference = %15.10f\n", EI_LS- EI0);
#endif
					EI0 = EI_LS;
					break;
				}
				else{

					stepsize_EI = stepsize_EI * 0.5;
					dvGradientSearch = dvGradientSearchSave;
#if 0
					printf("stepsize_EI = %15.10f\n",stepsize_EI);

#endif
					if(stepsize_EI < 10E-12) {
#if 0
						printf("The stepsize is getting too small!\n");
#endif

						breakOptimization = true;
						break;
					}
				}

			}
#if 0
			printf("dvGradientSearch:\n");
			dvGradientSearch.print();
			printf("EI0 = %15.10f\n",EI0);

#endif

			if(breakOptimization) break;

		} /* end of gradient-search loop */

		best_dv = dvGradientSearch;

#if 0
		printf("The most promising design:\n");
		best_dv.print();
#endif



		rowvec best_dvNorm = best_dv;

		best_dv =normalizeRowVectorBack(best_dvNorm, lowerBounds,upperBounds);


#if 0
		printf("The most promising design (not normalized):\n");
		best_dv.print();
#endif



		/* now make a simulation for the most promising design */

		double fVal = objFun.evaluate(best_dv);

		rowvec constraintValues(constraintFunctions.size());

		if(constraintFunctions.size() > 0){


			evaluateConstraints(best_dv,constraintValues);

			bool ifConstraintsSatisfied = checkConstraintFeasibility(constraintValues);

			if(!ifConstraintsSatisfied){

				fVal = LARGE;
			}

		}

		unsigned int numberOfEntries = dimension+1+constraintFunctions.size();
		optimizationHistory.resize(optimizationHistory.n_rows+1, numberOfEntries);


		/* add new values to the optimization History */
		for(unsigned int k=0; k<dimension;k++){

			optimizationHistory(optimizationHistory.n_rows-1,k) = best_dv(k);

		}

		optimizationHistory(optimizationHistory.n_rows-1,dimension) = fVal;

		for(unsigned int k=dimension+1; k<numberOfEntries;k++){

			optimizationHistory(optimizationHistory.n_rows-1,k) = constraintValues(k-dimension-1);

		}

		optimizationHistory.save("optimizationHistory.csv",csv_ascii);


#if 0
		printf("optimizationHistory:\n");
		optimizationHistory.print();

#endif



		double objFunVal = fVal;

		if(objFunVal < bestObjFunVal){

			bestIndx = iterOpt;
			bestObjFunVal = objFunVal;
			best_dvGlobal = best_dv;
#if 0
			printf("\nBetter design has been found:\n");
			printf("dv =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",objFunVal);
#endif

		}


		simulationCount ++;
#if 1
		//		printf("Simulation at dv = \n");
		best_dv.print();
		//		printf("True value of the function = %15.10f\n",fVal);

#endif



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


void Optimizer::performDoE(unsigned int howManySamples, DoE_METHOD methodID){

	if(!ifBoxConstraintsSet){

		cout<<"ERROR: Cannot run DoE before the box-constraints are set!\n";
		abort();
	}

	mat samples;

	if(methodID == LHS){

		LHSSamples DoE(dimension, lowerBounds, upperBounds, howManySamples);
		if(dimension == 2) DoE.visualize();
		std::string filename= this->name + "_samples.csv";
		DoE.saveSamplesToFile(filename);
		samples = DoE.getSamples();
	}

#if 1
	printMatrix(samples,"samples");
#endif

	mat bufferDataObjFunction(howManySamples,dimension+1);
	cube bufferDataConstraints(howManySamples,dimension+1,numberOfConstraints);


	rowvec constraintValues(numberOfConstraints);
	for(unsigned int i=0; i<howManySamples; i++){

		rowvec designVector = samples.row(i);


		for(unsigned int j=0;j<dimension;j++) {
			bufferDataObjFunction(i,j) = samples(i,j);

		}
		bufferDataObjFunction(i,dimension) = objFun.evaluate(designVector, false);

		evaluateConstraints(designVector, constraintValues,false);

		for(unsigned int j=0;j<dimension;j++) {
			for(unsigned int k=0;k<numberOfConstraints;k++) {

				bufferDataConstraints(i,j,k) = samples(i,j);

			}

		}

		for(unsigned int k=0;k<numberOfConstraints;k++) {

			bufferDataConstraints(i,dimension,k) = constraintValues(k);

		}


	}

	objFun.saveDoEData(bufferDataObjFunction);


	unsigned int contraintIt = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){


		if(it->ifNeedsSurrogate){


			mat saveBuffer = bufferDataConstraints.slice(contraintIt);
			it->saveDoEData(saveBuffer);
		}

		contraintIt++;
	}




}
void testOptimizationWingweight(void){
/*
	Sw: Wing Area (ft^2) (150,200)
	 *  Wfw: Weight of fuel in the wing (lb) (220,300)
	 *  A: Aspect ratio (6,10)
	 *  Lambda: quarter chord sweep (deg) (-10,10)
	 *  q: dynamic pressure at cruise (lb/ft^2)  (16,45)
	 *  lambda: taper ratio (0.5,1)
	 *  tc: aerofoil thickness to chord ratio (0.08,0.18)
	 *  Nz: ultimate load factor (2.5,6)
	 *  Wdg: flight design gross weight (lb)  (1700,2500)
	 *  Wp: paint weight (lb/ft^2) (0.025, 0.08)


*/
	vec lb(10);
	vec ub(10);
	lb(0) = 150; ub(0) = 200;
	lb(1) = 220; ub(1) = 300;
	lb(2) = 6;   ub(2) = 10;
	lb(3) = -10; ub(3) = 10;
	lb(4) = 16; ub(4) = 45;
	lb(5) = 0.5; ub(5) = 1;
	lb(6) = 0.08; ub(6) = 0.18;
	lb(7) = 2.5; ub(7) = 6;
	lb(8) = 1700; ub(8) = 2500;
	lb(9) = 0.025; ub(9) = 0.08;



	std::string problemName = "Wingweight";
	unsigned int dimension = 10;
	Optimizer OptimizationStudy(problemName, dimension);

	ObjectiveFunction objFunc(problemName, Wingweight, dimension);
	OptimizationStudy.addObjectFunction(objFunc);

	OptimizationStudy.maxNumberOfSamples = 100;

	OptimizationStudy.setBoxConstraints(lb,ub);
	OptimizationStudy.performDoE(100,LHS);


	OptimizationStudy.EfficientGlobalOptimization();


}

void testOptimizationEggholder(void){


	std::string problemName = "Eggholder";
	unsigned int dimension = 2;
	Optimizer OptimizationStudy(problemName, dimension);

	ObjectiveFunction objFunc(problemName, Eggholder, dimension);
	OptimizationStudy.addObjectFunction(objFunc);

	OptimizationStudy.maxNumberOfSamples = 50;

	OptimizationStudy.setBoxConstraints(0.0,200.0);
	OptimizationStudy.performDoE(50,LHS);

	OptimizationStudy.ifVisualize = true;
	OptimizationStudy.EfficientGlobalOptimization();


}

void testOptimizationHimmelblau(void){


	std::string problemName = "Himmelblau";
	unsigned int dimension = 2;
	Optimizer OptimizationStudy(problemName, dimension);

	ObjectiveFunction objFunc(problemName, Himmelblau, dimension);
	OptimizationStudy.addObjectFunction(objFunc);

	OptimizationStudy.maxNumberOfSamples = 100;

	OptimizationStudy.setBoxConstraints(-5.0,5.0);
	OptimizationStudy.performDoE(50,LHS);

	OptimizationStudy.ifVisualize = true;
	OptimizationStudy.EfficientGlobalOptimization();


}

void testOptimizationHimmelblauExternalExe(void){


	std::string problemName = "Himmelblau";
	unsigned int dimension = 2;
	Optimizer OptimizationStudy(problemName, dimension);

	ObjectiveFunction objFunc(problemName, dimension);
	objFunc.setFileNameReadObjectFunction("objFunVal.dat");
	objFunc.setExecutablePath("/home/emre/RoDeO/Tests/HimmelblauOptimization");
	objFunc.setExecutableName("himmelblau");
	objFunc.setFileNameDesignVector("dv.csv");
	OptimizationStudy.addObjectFunction(objFunc);

	OptimizationStudy.maxNumberOfSamples = 100;

	OptimizationStudy.setBoxConstraints(-5.0,5.0);
	OptimizationStudy.performDoE(50,LHS);

	OptimizationStudy.ifVisualize = true;
	OptimizationStudy.EfficientGlobalOptimization();


}




