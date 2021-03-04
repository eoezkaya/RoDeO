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
#ifndef DESIGN_HPP
#define DESIGN_HPP

#include <armadillo>
#include <vector>
#include "random_functions.hpp"
using namespace arma;

class CDesignExpectedImprovement{

public:
	rowvec dv;
	double valueExpectedImprovement;
	double objectiveFunctionValue;
	rowvec constraintValues;
	unsigned int dim;

	CDesignExpectedImprovement(unsigned int dimension, unsigned int numberOfConstraints){

		dim = dimension;
		constraintValues = zeros<rowvec>(numberOfConstraints);
		valueExpectedImprovement = 0.0;
		objectiveFunctionValue = 0.0;

	}

	CDesignExpectedImprovement(rowvec designVector, unsigned int numberOfConstraints){

			dv = designVector;
			constraintValues = zeros<rowvec>(numberOfConstraints);
			valueExpectedImprovement = 0.0;
			objectiveFunctionValue = 0.0;

	}

	void generateRandomDesignVector(void){

		dv = generateRandomRowVector(0.0,1.0/dim, dim);

	}

	void print(void) const{

		std::cout<<"Design vector = \n";
		dv.print();
		std::cout<<"Objective function value ="<<objectiveFunctionValue<<"\n";
		std::cout<<"Expected Improvement value = "<<valueExpectedImprovement<<"\n";
		std::cout<<"Constraint values = \n";
		constraintValues.print();
	}


};




class Design{

public:

	unsigned int dimension;
	unsigned int numberOfConstraints;
	rowvec designParameters;
	rowvec constraintTrueValues;
	rowvec gradient;
	double trueValue;
	double objectiveFunctionValue;
	std::vector<rowvec> constraintGradients;

	Design(rowvec);
	Design(unsigned int);
	void setNumberOfConstraints(unsigned int howManyConstraints);
	void print(void) const;
	void saveDesignVectorAsCSVFile(std::string fileName) const;
	void saveDesignVector(std::string fileName) const;
	void generateRandomDesignVector(vec lb, vec ub);
	void generateRandomDesignVector(double lb, double ub);
	bool checkIfHasNan(void) const;

	rowvec constructSampleObjectiveFunction(void) const;
	rowvec constructSampleObjectiveFunctionWithGradient(void) const;

	rowvec constructSampleConstraint(unsigned int constraintID) const;
	rowvec constructSampleConstraintWithGradient(unsigned int constraintID) const;

};




#endif
