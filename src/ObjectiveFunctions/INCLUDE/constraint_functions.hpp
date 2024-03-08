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
#ifndef CONSTRAINT_FUNCTIONS_HPP
#define CONSTRAINT_FUNCTIONS_HPP


#include <armadillo>
#include <cassert>

#include "./objective_function.hpp"
#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"
#include "../../../externalFunctions/INCLUDE/externalFunctions.hpp"

typedef double (*FunctionPtr)(double*);

class ConstraintDefinition{

public:

	std::string inequalityType;
	std::string constraintName;
	int ID = -1;
	double value = 0.0;


	void setDefinition(std::string definition);
	void print(void) const;

};



class ConstraintFunction: public ObjectiveFunction {

#ifdef UNIT_TESTS
	friend class ConstraintFunctionTest;
	FRIEND_TEST(ConstraintFunctionTest, useFunctionPointer);

#endif

private:

	ConstraintDefinition definitionConstraint;

	bool ifFunctionExplictlyDefined = false;
	double (*functionPtr)(double*) = NULL;


	std::vector<FunctionPtr> functionVector = { constraintFunction0,
			constraintFunction1,
			constraintFunction2};


public:

	ConstraintFunction();

	double interpolate(rowvec x) const;
	pair<double, double> interpolateWithVariance(rowvec x) const;

	void setConstraintDefinition(ConstraintDefinition);

	void setInequalityType(std::string);
	std::string getInequalityType(void) const;

	void setInequalityTargetValue(double);
	double getInequalityTargetValue(void) const;

	bool checkFeasibility(double value) const;

	void setID(int givenID);
	int getID(void) const;

	void readOutputDesign(Design &d) const;

	void evaluateDesign(Design &d);
	void addDesignToData(Design &d);

	void print(void) const;

	void setUseExplicitFunctionOn(void);

};



#endif
