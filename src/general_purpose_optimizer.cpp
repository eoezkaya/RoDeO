
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


#include "general_purpose_optimizer.hpp"
#include "LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "random_functions.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<cassert>

using namespace arma;
using namespace std;

unsigned int GeneralPurposeOptimizer::getDimension(void) const{

	return dimension;

}


void GeneralPurposeOptimizer::setDimension(unsigned int dim){

	dimension = dim;
	parameterBounds.setDimension(dim);

}


void GeneralPurposeOptimizer::setBounds(double lb, double ub){

	assert(dimension>0);
	parameterBounds.setBounds(lb,ub);

}

void GeneralPurposeOptimizer::setBounds(vec lb, vec ub){

	assert(lb.size() == ub.size());
	assert(lb.size() == dimension);
	parameterBounds.setBounds(lb,ub);

}

void GeneralPurposeOptimizer::setBounds(Bounds input){

	assert(input.areBoundsSet());
	parameterBounds = input;

}


void GeneralPurposeOptimizer::setDisplayOn(void){

	output.ifScreenDisplay = true;
}
void GeneralPurposeOptimizer::setDisplayOff(void){

	output.ifScreenDisplay = false;
}

void GeneralPurposeOptimizer::setMaxNumberOfFunctionEvaluations(unsigned int nMax){

	maxNumberOfFunctionEvaluations = nMax;

}

void GeneralPurposeOptimizer::setProblemName(std::string name){

	assert(name.empty() == false);
	problemName = name;
	ifProblemNameIsSet = true;

}


void GeneralPurposeOptimizer::setFilenameOptimizationHistory(std::string name){
	assert(name.empty() == false);
	filenameOptimizationHistory = name;
	ifFilenameOptimizationHistoryIsSet = true;

}
void GeneralPurposeOptimizer::setFilenameWarmStart(std::string name){

	assert(name.empty() == false);
	filenameWarmStart = name;
	ifFilenameWarmStartIsSet = true;
}
void GeneralPurposeOptimizer::setFilenameOptimizationResult(std::string name){

	assert(name.empty() == false);
	filenameOptimizationResult = name;
	ifFilenameOptimizationResultIsSet = true;
}



bool GeneralPurposeOptimizer::isProblemNameSet(void){
	return ifProblemNameIsSet;
}
bool GeneralPurposeOptimizer::isFilenameOptimizationHistorySet(void){
	return ifFilenameOptimizationHistoryIsSet;
}
bool GeneralPurposeOptimizer::isFilenameWarmStartSet(void){
	return ifFilenameWarmStartIsSet;
}
bool GeneralPurposeOptimizer::isFilenameOptimizationResultSet(void){
	return ifFilenameOptimizationResultIsSet;
}




void GeneralPurposeOptimizer::optimize(void){

	assert(false);

}


bool GeneralPurposeOptimizer::isObjectiveFunctionSet(void) const{

	return ifObjectiveFunctionIsSet;

}

bool GeneralPurposeOptimizer::areBoundsSet(void) const {

	return parameterBounds.areBoundsSet();

}


void GeneralPurposeOptimizer::setObjectiveFunction(GeneralPurposeObjectiveFunction functionToSet ){

	assert(functionToSet != NULL);
	calculateObjectiveFunction = functionToSet;
	ifObjectiveFunctionIsSet = true;

}


void GeneralPurposeOptimizer::setNumberOfThreads(unsigned int nThreads){

	numberOfThreads = nThreads;
	omp_set_num_threads(numberOfThreads);


}

unsigned int GeneralPurposeOptimizer::getNumberOfThreads(void) const{

	return numberOfThreads;

}


double GeneralPurposeOptimizer::callObjectiveFunction(vec &x){

	if(isObjectiveFunctionSet()){

		return calculateObjectiveFunction(x);

	}
	else{
		return calculateObjectiveFunctionInternal(x);

	}

}

double GeneralPurposeOptimizer::calculateObjectiveFunctionInternal(vec& x){

	double someNumber = -19.12;
	return someNumber;

}

void GeneralPurposeOptimizer::writeWarmRestartFile(void){

	assert(false);

}

