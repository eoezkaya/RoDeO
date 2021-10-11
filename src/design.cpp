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


#include <cassert>
#include "design.hpp"
#include "matrix_vector_operations.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


Design::Design(rowvec dv){

	dimension = dv.size();
	designParameters = dv;
	gradient = zeros<rowvec>(dimension);


}

Design::Design(unsigned int dim){

	dimension = dim;
	designParameters = zeros<rowvec>(dimension);
	gradient = zeros<rowvec>(dimension);


}

void Design::setNumberOfConstraints(unsigned int howManyConstraints){

	numberOfConstraints = howManyConstraints;
	constraintTrueValues = zeros<rowvec>(numberOfConstraints);


}


void Design::generateRandomDesignVector(vec lb, vec ub){

	designParameters = generateRandomRowVector(lb,ub);


}

void Design::generateRandomDesignVector(double lb, double ub){

	designParameters = generateRandomRowVector(lb,ub,dimension);


}

rowvec Design::constructSampleObjectiveFunction(void) const{

	rowvec sample(dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		sample(i) = designParameters(i);
	}

	sample(dimension) = trueValue;


	return sample;
}

rowvec Design::constructSampleObjectiveFunctionWithGradient(void) const{

	rowvec sample(2*dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		sample(i) = designParameters(i);
	}

	sample(dimension) = trueValue;

	for(unsigned int i=0; i<dimension; i++){


		sample(dimension+1+i) = gradient(i);

	}

	return sample;
}

bool Design::checkIfHasNan(void) const{

	bool ifHasNan = false;
	if(std::isnan(trueValue )) {

		ifHasNan = true;
	}
	if(gradient.has_nan()){

		ifHasNan = true;
	}

	if(constraintTrueValues.has_nan()){

		ifHasNan = true;
	}

	for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

		if(it->has_nan()){

			ifHasNan = true;
		}


	}
	return ifHasNan;


}

rowvec Design::constructSampleConstraint(unsigned int constraintID) const{

	rowvec sample(dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		sample(i) = designParameters(i);
	}

	sample(dimension) = constraintTrueValues(constraintID);


	return sample;
}

rowvec Design::constructSampleConstraintWithGradient(unsigned int constraintID) const{

	rowvec sample(2*dimension+1);

	for(unsigned int i=0; i<dimension; i++){

		sample(i) = designParameters(i);
	}

	sample(dimension) = constraintTrueValues(constraintID);

	rowvec constraintGradient = constraintGradients.at(constraintID);
	for(unsigned int i=0; i<dimension; i++){


		sample(dimension+1+i) = constraintGradient(i);

	}


	return sample;
}


Design::Design(void){


}

void Design::print(void) const{


	std::cout<<"\n\nPrinting Design...\n";
	std::cout<<"Design parameters = \n";
	printVector(designParameters);
	std::cout<<"Function value = "<<trueValue<<"\n";


	if(!gradient.is_zero() && gradient.size() > 0){

		printVector(gradient,"gradient vector");

	}



	if(constraintTrueValues.size() >0){

		printVector(constraintTrueValues,"constraint values");

	}




	if(!constraintGradients.empty()){

		int count = 0;
		for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

			if(!it->is_zero()){
				std::cout<<"Constraint gradient "<<count<<"\n";
				printVector(*it);
				count++;
			}


		}
	}

	std::cout<<"Objective function value = "<<objectiveFunctionValue<<"\n";
	std::cout<<"Improvement = "<<improvementValue<<"\n";


}



void Design::saveToAFile(std::string filename) const{

	assert(!filename.empty());
	std::ofstream fileOut;
	fileOut.open (filename);
	fileOut << "Tag: "<<tag<<"\n";
	fileOut << "Design parameters vector:\n";
	fileOut << designParameters;
	fileOut << "Objective function value: "<<objectiveFunctionValue<<"\n";

	if(numberOfConstraints>0){

		fileOut << "Constraint values vector:\n";
		fileOut << constraintTrueValues;

	}


	fileOut.close();




}

void Design::saveDesignVectorAsCSVFile(std::string fileName) const{

	std::ofstream designVectorFile (fileName);
	designVectorFile.precision(10);
	if (designVectorFile.is_open())
	{

		for(unsigned int i=0; i<designParameters.size()-1; i++){

			designVectorFile << designParameters(i)<<",";
		}

		designVectorFile << designParameters(designParameters.size()-1);

		designVectorFile.close();
	}
	else{

		cout << "ERROR: Unable to open file";
		abort();
	}
}

void Design::saveDesignVector(std::string fileName) const{

	std::ofstream designVectorFile (fileName);
	designVectorFile.precision(10);
	if (designVectorFile.is_open())
	{

		for(unsigned int i=0; i<designParameters.size()-1; i++){

			designVectorFile << designParameters(i)<<" ";
		}

		designVectorFile << designParameters(designParameters.size()-1);

		designVectorFile.close();
	}
	else{

		cout << "ERROR: Unable to open file";
		abort();
	}
}


void CDesignExpectedImprovement::generateRandomDesignVector(void){

	double lowerBound = 0.0;
	double upperBound = 1.0/dim;
	dv = generateRandomRowVector(lowerBound, upperBound , dim);


}

void CDesignExpectedImprovement::generateRandomDesignVector(vec lb, vec ub){


	dv = generateRandomRowVector(lb, ub);


}


void CDesignExpectedImprovement::generateRandomDesignVectorAroundASample(const rowvec &sample, vec lb, vec ub){


	vec lowerBounds(dim);
	vec upperBounds(dim);

	double factor = 1.0/dim;
	double dx = factor*0.01;


	for(unsigned int i=0; i<dim; i++){

		lowerBounds(i) = sample(i) - dx;
		upperBounds(i) = sample(i) + dx;
		if(lowerBounds(i) < lb(i))    lowerBounds(i) = lb(i);
		if(upperBounds(i) > lb(i)) upperBounds(i) = ub(i);

	}

	dv = generateRandomRowVector(lowerBounds, upperBounds);


}
