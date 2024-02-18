/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#include <cassert>
#include "./INCLUDE/optimization_history.hpp"
#include "./INCLUDE/design.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../Auxiliary/INCLUDE/xml_functions.hpp"
#include "../LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"




#include <armadillo>
using namespace arma;


void OptimizationHistory::addConstraintName(string name){

	assert(isNotEmpty(name));
	constraintNames.push_back(name);

}


void OptimizationHistory::setDimension(unsigned int dim){

	assert(dim>0);
	dimension = dim;
}

void OptimizationHistory::setData(mat dataIn){

	assert(dataIn.n_rows>0);
	unsigned int numberOfEntries = dimension + 1 + constraintNames.size() + 2;
	assert(dataIn.n_cols == numberOfEntries);
	data = dataIn;

}

mat OptimizationHistory::getData(void) const{
	return data;
}

vec OptimizationHistory::getObjectiveFunctionValues(void) const{
	assert(dimension>0);
	return data.col(dimension);
}

vec OptimizationHistory::getFeasibilityValues(void) const{
	assert(data.n_cols>0);
	return data.col(data.n_cols-1);

}

void OptimizationHistory::setObjectiveFunctionName(string name){

	assert(isNotEmpty(name));
	objectiveFunctionName = name;

}

field<std::string> OptimizationHistory::setHeader(void) const{

	field<std::string> fileHeader(dimension + 1 + constraintNames.size() + 2);
	bool ifVariableNamesAreSet = false;

	if(variableNames.size()>0){

		assert(variableNames.size() == dimension);
		ifVariableNamesAreSet = true;
	}

	unsigned int count = 0;

	if(ifVariableNamesAreSet){

		for (auto it = variableNames.begin();it != variableNames.end(); it++) {

			fileHeader(count) = *it;
			count++;

		}

	}

	else{

		for(unsigned int i=0; i<dimension; i++){
			string variable = "x" + std::to_string(i+1);
			fileHeader(count) = variable;
			count++;

		}
	}

	assert(isNotEmpty(objectiveFunctionName));
	fileHeader(count) = objectiveFunctionName;
	count++;


	for (auto it = constraintNames.begin();it !=constraintNames.end(); it++) {

		fileHeader(count) = *it;
		count++;

	}
	fileHeader(count) = "Improvement";
	count++;
	fileHeader(count) = "Feasibility";
	count++;

	return fileHeader;

}

void OptimizationHistory::saveOptimizationHistoryFile(void){

	assert(!data.empty());
	field<std::string> fileHeader = setHeader();
	assert(fileHeader.n_elem == data.n_cols);

	data.save( csv_name(filename, fileHeader) );

}

void OptimizationHistory::updateOptimizationHistory(Design d) {

	unsigned int numberOfEntries = dimension + 1 + constraintNames.size() + 2;
	assert(d.designParameters.size() == dimension);

	rowvec newRow(numberOfEntries);

	for(unsigned int i=0; i<dimension; i++) {
		newRow(i) = d.designParameters(i);
	}

	newRow(dimension) = d.trueValue;

	for(unsigned int i=0; i<constraintNames.size(); i++){
		newRow(i+dimension+1) = 	d.constraintTrueValues(i);
	}

	newRow(dimension + constraintNames.size()+1)   = d.improvementValue;

	if(d.isDesignFeasible){
		newRow(dimension + constraintNames.size()+2) = 1.0;
	}
	else{
		newRow(dimension + constraintNames.size()+2) = 0.0;
	}

	data.insert_rows( data.n_rows, newRow );
	saveOptimizationHistoryFile();

}


double OptimizationHistory::calculateInitialImprovementValue(void) const{

	unsigned int N = data.n_rows;
	assert(N>0);

	vec objectiveFunctionValues = getObjectiveFunctionValues();
	vec feasibilityValues = getFeasibilityValues();

	bool ifFeasibleDesignFound = false;
	double bestFeasibleObjectiveFunctionValue = LARGE;

	for(unsigned int i=0; i<N; i++){

		if(feasibilityValues(i) > 0.0 && objectiveFunctionValues(i) < bestFeasibleObjectiveFunctionValue){
			ifFeasibleDesignFound = true;
			bestFeasibleObjectiveFunctionValue = objectiveFunctionValues(i);

		}
	}

	if(ifFeasibleDesignFound){
		return bestFeasibleObjectiveFunctionValue;
	}
	else return 0.0;

}


void OptimizationHistory::print(void) const{

	std::cout<<"dimension: "<<dimension<<"\n";
	std::cout<<"objective function name: "<<objectiveFunctionName<<"\n";

	if(constraintNames.size()>0){
		std::cout<<"constraint names = \n";
		for (auto it = constraintNames.begin();it !=constraintNames.end(); it++) {

			std::cout<<*it<<"\n";
		}
	}

}

