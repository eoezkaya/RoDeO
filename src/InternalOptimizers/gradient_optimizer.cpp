#include "./INCLUDE/gradient_optimizer.hpp"
//#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"


#include<cassert>

using namespace std;


namespace Rodop{

GradientOptimizer::GradientOptimizer(){

}




bool  GradientOptimizer::isInitialPointSet(void) const{

	return ifInitialPointIsSet;

}


bool  GradientOptimizer::isGradientFunctionSet(void) const{

	return ifGradientFunctionIsSet;

}




void GradientOptimizer::setInitialPoint(const vec &input){

	assert(input.getSize() == getDimension());
	currentIterate.x = input;

	ifInitialPointIsSet = true;

}



void GradientOptimizer::setMaximumStepSize(double value){

	maximumStepSize = value;
}
void GradientOptimizer::setTolerance(double value){

	tolerance = value;
}
void GradientOptimizer::setMaximumNumberOfIterationsInLineSearch(int value){

	maximumNumberOfIterationsInLineSearch = value;

}




void GradientOptimizer::setGradientFunction(GradientFunctionType functionToSet ){

	assert(functionToSet != NULL);

	calculateGradientFunction = functionToSet;

	ifGradientFunctionIsSet = true;


}



void GradientOptimizer::setMaximumNumberOfFunctionEvaluations(int input){

	numberOfMaximumFunctionEvaluations = input;

}

void GradientOptimizer::setFiniteDifferenceMethod(string method){

	assert(method == "central" || method == "forward");
	finiteDifferenceMethod = method;
	areFiniteDifferenceApproximationsToBeUsed = true;


}

void GradientOptimizer::checkIfOptimizationSettingsAreOk(void) const{


	assert(dimension >0);
	assert(ifInitialPointIsSet);
	assert(numberOfMaximumFunctionEvaluations > 0);
	assert(areBoundsSet());

}

void GradientOptimizer::setEpsilonForFiniteDifference(double value) {

	assert(value>0.0);
	assert(value<0.1);
	epsilonForFiniteDifferences = value;

}



vec GradientOptimizer::calculateGradientFunctionInternal(const vec &input){

	assert(ifGradientFunctionIsSet);

	double *gradient = new double[dimension];
	calculateGradientFunction(input.getPointer(), gradient);

	vec result(dimension);

	for(unsigned int i=0; i<dimension; i++) result(i) = gradient[i];
	delete[] gradient;

	return result;
}

vec GradientOptimizer::calculateCentralFiniteDifferences(designPoint &input) {

	vec gradient(dimension);

	for (unsigned int i = 0; i < dimension; i++) {
		double inputSave = input.x(i);
		double eps = epsilonForFiniteDifferences * input.x(i);
		input.x(i) += eps;
		evaluateObjectiveFunction(input);
		double fplus = input.objectiveFunctionValue;
		input.x(i) -= 2 * eps;
		evaluateObjectiveFunction(input);
		double fminus = input.objectiveFunctionValue;
		input.x(i) = inputSave;
		gradient(i) = (fplus - fminus) / (2 * eps);
	}

	return gradient;
}

vec GradientOptimizer::calculateForwardFiniteDifferences(designPoint &input) {

	vec gradient(dimension);
	evaluateObjectiveFunction(input);
	double f0 = input.objectiveFunctionValue;

	for (unsigned int i = 0; i < dimension; i++) {
		double inputSave = input.x(i);
		double eps = epsilonForFiniteDifferences * input.x(i);
		input.x(i) += eps;
		evaluateObjectiveFunction(input);
		double fplus = input.objectiveFunctionValue;
		input.x(i) = inputSave;
		gradient(i) = (fplus - f0) / (eps);
	}
	return gradient;
}

void GradientOptimizer::approximateGradientUsingFiniteDifferences(designPoint &input){

	assert(dimension>0);



	if(finiteDifferenceMethod == "central"){

		input.gradient = calculateCentralFiniteDifferences(input);


	}

	if(finiteDifferenceMethod == "forward"){

		calculateForwardFiniteDifferences(input);
	}


}

void GradientOptimizer::evaluateObjectiveFunction(designPoint &input){

	double objectiveFunctionValue = 0.0;

	if(ifObjectiveFunctionIsSet){

		objectiveFunctionValue = calculateObjectiveFunction(input.x.getPointer());

	}
	else{

		objectiveFunctionValue = calculateObjectiveFunctionInternal(input.x);

	}

	input.objectiveFunctionValue = objectiveFunctionValue;
	numberOfFunctionEvaluations++;


}


void GradientOptimizer::evaluateGradientFunction(designPoint &input){

	if(areFiniteDifferenceApproximationsToBeUsed){

		approximateGradientUsingFiniteDifferences(input);

	}


	else if(ifGradientFunctionIsSet){

		double *gradient = new double[dimension];
		calculateGradientFunction(input.x.getPointer(), gradient);

		input.gradient.resize(dimension);

		for(unsigned int i=0; i<dimension; i++) input.gradient(i) = gradient[i];
		delete[] gradient;


	}
	else{

		input.gradient = calculateGradientFunctionInternal(input.x);
	}

	input.L2NormGradient = input.gradient.norm();

}


void GradientOptimizer::performGoldenSectionSearch(void){





}

void GradientOptimizer::performBacktrackingLineSearch(void){

	assert(maximumStepSize > 0);

	designPoint currentIterateSave = currentIterate;

	double stepSize = maximumStepSize;
	while(true){


		vec deltax = currentIterate.gradient*stepSize;

		nextIterate.x = currentIterate.x - deltax;

		evaluateObjectiveFunction(nextIterate);

		if(nextIterate.objectiveFunctionValue < currentIterate.objectiveFunctionValue){

			break;
		}
		else{

//			output.printMessage("Decreasing stepsize...");
			stepSize = stepSize*0.5;
//			output.printMessage("Stepsize = " , stepSize);
		}



	}


	currentIterate = nextIterate;
	evaluateGradientFunction(currentIterate);

//	output.printMessage("f1 = ",currentIterate.objectiveFunctionValue);
	optimalObjectiveFunctionValue = currentIterate.objectiveFunctionValue;



}



void GradientOptimizer::performLineSearch(void){

	if(LineSearchMethod == "golden_section_search"){

		performGoldenSectionSearch();
	}

	if(LineSearchMethod == "backtracking_line_search"){

		performBacktrackingLineSearch();

	}


}


void GradientOptimizer::optimize(void){


	checkIfOptimizationSettingsAreOk();

	evaluateObjectiveFunction(currentIterate);
	evaluateGradientFunction(currentIterate);

//	output.printMessage("Initial gradient", currentIterate.gradient);

	unsigned int noOptimizationIteration = 1;
	while(true){


		if(currentIterate.L2NormGradient < tolerance){

			break;
		}

		performLineSearch();





		if(numberOfFunctionEvaluations >= numberOfMaximumFunctionEvaluations){

			break;
		}

	}




	noOptimizationIteration++;
}


double GradientOptimizer::getOptimalObjectiveFunctionValue(void) const{

	return optimalObjectiveFunctionValue;
}


} /* Namespace Rodop */
