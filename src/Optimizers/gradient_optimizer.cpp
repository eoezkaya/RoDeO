/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 *  file is part of RoDeO
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


#include "gradient_optimizer.hpp"
#include "LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<cassert>

using namespace arma;
using namespace std;


GradientOptimizer::GradientOptimizer(){

}




bool  GradientOptimizer::isInitialPointSet(void) const{

	return ifInitialPointIsSet;

}


bool  GradientOptimizer::isGradientFunctionSet(void) const{

	return ifGradientFunctionIsSet;

}




void GradientOptimizer::setInitialPoint(vec input){

	assert(input.size() == getDimension());
	currentIterate.x = input;

	ifInitialPointIsSet = true;

}



void GradientOptimizer::setMaximumStepSize(double value){

	maximumStepSize = value;
}
void GradientOptimizer::setTolerance(double value){

	tolerance = value;
}
void GradientOptimizer::setMaximumNumberOfIterationsInLineSearch(unsigned int value){

	maximumNumberOfIterationsInLineSearch = value;

}




void GradientOptimizer::setGradientFunction(GradientFunctionType functionToSet ){

	assert(functionToSet != NULL);

	calculateGradientFunction = functionToSet;

	ifGradientFunctionIsSet = true;


}



void GradientOptimizer::setMaximumNumberOfFunctionEvaluations(unsigned int input){

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



vec GradientOptimizer::calculateGradientFunctionInternal(vec input){

	assert(ifGradientFunctionIsSet);
	return calculateGradientFunction(input);


}

vec GradientOptimizer::calculateCentralFiniteDifferences(designPoint &input) {

	vec gradient(dimension,fill::zeros);

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
	vec gradient(dimension, fill::zeros);
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

		objectiveFunctionValue = calculateObjectiveFunction(input.x);

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

		input.gradient = calculateGradientFunction(input.x);

	}
	else{

		input.gradient = calculateGradientFunctionInternal(input.x);
	}

	input.L2NormGradient = norm(input.gradient);



}


void GradientOptimizer::performGoldenSectionSearch(void){





}

void GradientOptimizer::performBacktrackingLineSearch(void){

	assert(maximumStepSize > 0);

	designPoint currentIterateSave = currentIterate;

	double stepSize = maximumStepSize;
	while(true){


		nextIterate.x = currentIterate.x - stepSize* currentIterate.gradient;

		evaluateObjectiveFunction(nextIterate);

		if(nextIterate.objectiveFunctionValue < currentIterate.objectiveFunctionValue){

			break;
		}
		else{

			output.printMessage("Decreasing stepsize...");
			stepSize = stepSize*0.5;
			output.printMessage("Stepsize = " , stepSize);
		}



	}


	currentIterate = nextIterate;
	evaluateGradientFunction(currentIterate);

	output.printMessage("f1 = ",currentIterate.objectiveFunctionValue);
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

	output.printMessage("Initial gradient", currentIterate.gradient);

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



