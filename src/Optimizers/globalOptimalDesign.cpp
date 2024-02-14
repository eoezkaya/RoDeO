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
#include "./INCLUDE/globalOptimalDesign.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../Auxiliary/INCLUDE/xml_functions.hpp"
#include "../LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"




#include <armadillo>
using namespace arma;


void GlobalOptimalDesign::setBoxConstraints(Bounds input){
	assert(input.areBoundsSet());
	assert(dimension == input.getDimension());

	boxConstraints = input;

}



void GlobalOptimalDesign::setGlobalOptimalDesignFromHistoryFile(const mat& historyFile){

	assert(historyFile.n_rows>0);
	assert(boxConstraints.areBoundsSet());
	assert(dimension>0);

	unsigned int howManyEntriesInHistoryFile = historyFile.n_cols;
	unsigned int howManySamplesInHistoryFile = historyFile.n_rows;

	unsigned int indexLastCol = historyFile.n_cols -1;

	bool isFeasibleDesignFound = false;
	double bestObjectiveFunctionValue = LARGE;
	unsigned int bestDesignIndex;

	for(unsigned int i=0; i<howManySamplesInHistoryFile; i++){

		double feasibility = historyFile(i,indexLastCol);
		double objectiveFunctionValue = historyFile(i,dimension);

		if(feasibility>0.0 && objectiveFunctionValue < bestObjectiveFunctionValue){
			isFeasibleDesignFound = true;
			bestObjectiveFunctionValue = objectiveFunctionValue;
			bestDesignIndex = i;
		}

	}

	rowvec bestSample;
	if(isFeasibleDesignFound){

		bestSample = historyFile.row(bestDesignIndex);

		isDesignFeasible = true;
		ID = bestDesignIndex;
	}

	else{

		vec objectiveFunctionValues = historyFile.col(dimension);

		uword indexMin = index_min(objectiveFunctionValues);
		bestSample = historyFile.row(indexMin);

		isDesignFeasible = false;
		ID = indexMin;
	}

	rowvec dv = bestSample.head(dimension);

	tag = "Global optimum design";
	designParameters  = dv;
	trueValue = bestSample(dimension);
	improvementValue = bestSample(historyFile.n_cols-2);

	rowvec constraintValues(numberOfConstraints);
	for(unsigned int i=0; i<numberOfConstraints; i++){

		constraintValues(i) = bestSample(i+dimension+1);
	}

	constraintTrueValues = constraintValues;

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	rowvec dvNormalized = normalizeVector(dv,lb,ub);
	designParametersNormalized = dvNormalized;

}

void GlobalOptimalDesign::setGradientGlobalOptimumFromTrainingData(const std::string &nameOfTrainingData){


	assert(isNotEmpty(nameOfTrainingData));
	assert(dimension>0);
	assert(designParameters.size() > 0);

	mat trainingDataToSearch;
	trainingDataToSearch.load(nameOfTrainingData, csv_ascii);


	mat trainingDataInput = trainingDataToSearch.submat(0,0,trainingDataToSearch.n_rows-1,dimension-1);

	int indexOfTheGlobalOptimalDesignInTrainingData = findIndexOfRow(designParameters, trainingDataInput,10E-06);

	if(indexOfTheGlobalOptimalDesignInTrainingData  == -1){

		cout<<"ERROR: Could not found the row in the training data!\n";
		designParameters.print("x");
		trainingDataInput.print("trainingDataInput");
		abort();
	}

	assert(indexOfTheGlobalOptimalDesignInTrainingData  != -1);
	assert(indexOfTheGlobalOptimalDesignInTrainingData < int(trainingDataToSearch.size()));

	rowvec temp = trainingDataToSearch.row(indexOfTheGlobalOptimalDesignInTrainingData);
	rowvec gradientVector = temp.tail(dimension);
	gradient = gradientVector;
}

void GlobalOptimalDesign::saveToXMLFile(void) const{


	assert(isNotEmpty(xmlFileName));
	std::ofstream file(xmlFileName);

	std::string text = generateXmlString();
	file << text;
	file.close();
}

std::string GlobalOptimalDesign::generateXmlString(void) const{

	std::string result;
	result = "<GlobalOptimalDesign>\n";
	result += generateXml("DesignID", ID) + "\n";
	result += generateXml("ObjectiveFunction", trueValue) + "\n";
	result += generateXmlVector("DesignParameters", designParameters) + "\n";
	if(constraintTrueValues.size() > 0){
		result += generateXmlVector("ConstraintValues", constraintTrueValues) + "\n";
	}

	if(isDesignFeasible){
		string yes = "YES";
		result += generateXml("Feasibility", yes) + "\n";
	}
	else{
		string no = "NO";
		result += generateXml("Feasibility", no) + "\n";
	}


	result += "</GlobalOptimalDesign>\n";

	return result;
}


bool GlobalOptimalDesign::checkIfGlobalOptimaHasGradientVector(void) const{

	if(gradient.empty()) {
		return false;
	}
	else{

		if(gradient.is_zero()){
			return false;
		}
		else{
			return true;
		}
	}
}
