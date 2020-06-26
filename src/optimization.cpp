#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxilliary_functions.hpp"
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
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


ObjectiveFunction::ObjectiveFunction(){

	dim = 0;
	name = "None";
	objectiveFunPtr = empty;

}

void ObjectiveFunction::trainSurrogate(void){

	surrogateModel.train();

}


double ObjectiveFunction::calculateExpectedImprovement(rowvec x){

	return surrogateModel.calculateExpectedImprovement(x);

}


double ObjectiveFunction::evaluate(rowvec x){

	double functionValue =  objectiveFunPtr(x.memptr());

	assert(std::isnan(functionValue) == false);

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
		return LARGE;

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


double ConstraintFunction::evaluate(rowvec x){

	double functionValue =  pConstFun(x.memptr());

	assert(std::isnan(functionValue) == false);

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

void OptimizerWithGradients::EfficientGlobalOptimization(void){

	print();

	AggregationModel ObjFunModel(name,size_of_dv);

	if(doesValidationFileExist){

		std::string validation_filename = name + "_validation.csv";
		ObjFunModel.validationset_input_filename = validation_filename;
		ObjFunModel.visualizeAggModelValidation = true;
		ObjFunModel.visualizeKrigingValidation = true;
		ObjFunModel.visualizeKernelRegressionValidation = false;
	}

	ObjFunModel.number_of_cv_iterations = 0;

	ObjFunModel.train();

	/* main loop for optimization */
	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;

	double bestObjFunVal = LARGE;
	rowvec best_dvGlobal(size_of_dv);
	unsigned int bestIndx = -1;

	while(1){
		iterOpt++;
#if 1
		printf("Optimization Iteration = %d\n",iterOpt);
		printf("Sample minimum = %10.7f\n",ObjFunModel.ymin );
#endif

		double maxEI = 0.0;
		rowvec best_dv(size_of_dv);

#pragma omp parallel for
		for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){
			rowvec dv(size_of_dv);
			rowvec dvNorm(size_of_dv);

			/* Generate a random design vector */
			for(unsigned int k=0; k<size_of_dv; k++){

				dv(k)= generateRandomDouble(lower_bound_dv(k), upper_bound_dv(k));
				dvNorm(k) = (1.0/ObjFunModel.dim)*(dv(k) - ObjFunModel.xmin(k)) / (ObjFunModel.xmax(k) - ObjFunModel.xmin(k));
			}

#if 0
			printf("dv = \n");
			dv.print();
#endif
			double ftilde = 0.0;
			double ssqr   = 0.0;
			ObjFunModel.ftilde_and_ssqr(dvNorm,&ftilde,&ssqr);


#if 0
			printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
#endif

			double	standart_error = sqrt(ssqr)	;

			double EI = 0.0;

			if(standart_error!=0.0){

				double	EIfac = (ObjFunModel.ymin - ftilde)/standart_error;

				/* calculate the Expected Improvement value */
				EI = (ObjFunModel.ymin - ftilde)*cdf(EIfac,0.0,1.0)+ standart_error * pdf(EIfac,0.0,1.0);
			}
			else{

				EI = 0.0;

			}
#if 0
			printf("EI value = %15.10f\n",EI);
#endif
			if(EI > maxEI){

				best_dv = dv;
				maxEI = EI;
#if 1
				printf("A design with better EI value has been find, EI = %15.10f\n", EI);
				best_dv.print();
#endif
			}


		} /* end of EI loop */

		/* now make a simulation for the most promising design */

		rowvec grad(size_of_dv);
		double fVal = adj_fun(best_dv.memptr(), grad.memptr());
		double objFunVal = fVal;

		if(objFunVal < bestObjFunVal){

			bestIndx = iterOpt;
			bestObjFunVal = objFunVal;
			best_dvGlobal = best_dv;
#if 1
			printf("\nBetter design has been found:\n");
			printf("dv =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",objFunVal);
#endif

		}


		simulationCount ++;
#if 1
		printf("Simulation at dv = \n");
		best_dv.print();
		printf("True value of the function = %15.10f\n",fVal);
		printf("grad = \n");
		grad.print();


#endif

		/* new row that will be added to the data */

		rowvec newsample(2*size_of_dv +1);

		for(unsigned int k=0; k<size_of_dv; k++){

			newsample(k) = best_dv(k);

		}
		newsample(size_of_dv) = fVal;

		for(unsigned int k=size_of_dv+1; k<2*size_of_dv+1; k++){

			newsample(k) = grad(k-size_of_dv-1);

		}
#if 1
		printf("new sample: \n");
		newsample.print();
#endif



		ObjFunModel.data.resize(ObjFunModel.N+1,2*ObjFunModel.dim+1);

		ObjFunModel.data.row(ObjFunModel.N) = newsample;
		ObjFunModel.data.save(ObjFunModel.input_filename,csv_ascii);
		ObjFunModel.update();


		if(simulationCount >= max_number_of_samples){

			printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");
			printf("Global optimal solution:\n");
			printf("dv =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",bestObjFunVal);
			printf("Index = %d\n",bestIndx);
			break;
		}



	}



}



Optimizer::Optimizer(std::string nameTestcase, int input_size,std::string problemType){

	/* we do not allow problems with too large dimension */

	if(input_size > 1000){

		std::cout<<"Problem dimension of the optimization is too large!"<<std::endl;
		exit(-1);

	}

	name = nameTestcase;
	dimension = input_size;
	numberOfConstraints = 0;
	maxNumberOfSamples  = 0;
	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);
	iterMaxEILoop = dimension*10000;
	iterGradientEILoop = 100;
	epsilon_EI = 10E-4;
	optimizationType = problemType;
	ifVisualize = false;
	howOftenTrainModels = 10;


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


}

void Optimizer::setBoxConstraints(double lowerBound, double upperBound){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	assert(lowerBound < upperBound);
	lowerBounds.fill(lowerBound);
	upperBounds.fill(upperBound);


}


void Optimizer::setBoxConstraints(vec lb, vec ub){

	std::cout<<"Setting box constraints for "<<name<<std::endl;
	for(unsigned int i=0; i<dimension; i++) assert(lb(i) < ub(i));

	lowerBounds = lb;
	upperBounds = ub;


}


void Optimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;

}


void Optimizer::addObjectFunction(ObjectiveFunction &objFunc){

	objFun = objFunc;

}


void Optimizer::evaluateConstraints(rowvec x, rowvec &constraintValues){

	unsigned int contraintIt = 0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		constraintValues(contraintIt) = it->evaluate(x);
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

		fprintf(stderr, "Error: maximum number of samples is not set!\n");
		exit(-1);
	}



	if(checkBoxConstraints() == false){

		fprintf(stderr, "Error: Box constraints are not set properly!\n");
		exit(-1);

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
#if 1
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
#if 1
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

#if 1
		printf("The most promising design:\n");
		best_dv.print();
#endif



		rowvec best_dvNorm = best_dv;

		best_dv =normalizeRowVectorBack(best_dvNorm, lowerBounds,upperBounds);


#if 1
		printf("The most promising design (not Normalized):\n");
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


#if 1
		printf("optimizationHistory:\n");
		optimizationHistory.print();

#endif



		double objFunVal = fVal;

		if(objFunVal < bestObjFunVal){

			bestIndx = iterOpt;
			bestObjFunVal = objFunVal;
			best_dvGlobal = best_dv;
#if 1
			printf("\nBetter design has been found:\n");
			printf("dv =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",objFunVal);
#endif

		}


		simulationCount ++;
#if 1
		printf("Simulation at dv = \n");
		best_dv.print();
		printf("True value of the function = %15.10f\n",fVal);

#endif


		/* terminate optimization */
		if(simulationCount >= maxNumberOfSamples){

			printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");
			printf("Global optimal solution:\n");
			printf("design vector =");
			best_dv.print();
			printf("Objective function value = %15.10f\n",bestObjFunVal);
			printf("Index = %d\n",bestIndx);


			if(ifVisualize){

				visualizeOptimizationHistory();

			}


			break;
		}



	} /* end of the optimization loop */



}



