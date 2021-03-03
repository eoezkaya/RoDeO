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
#ifndef CONSTRAINT_FUNCTIONS_HPP
#define CONSTRAINT_FUNCTIONS_HPP


#include <armadillo>
#include "objective_function.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "design.hpp"
#include <cassert>

class ConstraintFunction{


private:
	unsigned int ID = -1;
	unsigned int dim = 0;

	double (*pConstFun)(double *);
	double (*pConstFunAdj)(double *,double *);
	double targetValue  = 0.0;
	std::string inequalityType;

	KrigingModel surrogateModel;
	AggregationModel surrogateModelGradient;

	std::string executableName;
	std::string executablePath;
	std::string fileNameConstraintFunctionRead;
	std::string fileNameDesignVector;

	vec ub;
	vec lb;

	bool ifGradientAvailable = false;
	bool ifFunctionPointerIsSet = false;
	bool ifInitialized = false;
public:
	std::string name;

	std::vector<int> IDToFunctionsShareOutputExecutable;

	unsigned int readOutputStartIndex;


	ConstraintFunction(std::string, std::string, double, double (*constFun)(double *), unsigned int );
	ConstraintFunction(std::string, std::string, double, double (*constFun)(double *, double *), unsigned int );

	ConstraintFunction(std::string, std::string, double, unsigned int);
	ConstraintFunction();
	void saveDoEData(std::vector<rowvec> data) const;

	void initializeSurrogate(void);
	void trainSurrogate(void);


	void setGradientOn(void);
	void setGradientOff(void);
	void setParameterBounds(vec , vec );

	void setFileNameReadConstraintFunction(std::string);
	void setExecutablePath(std::string);
	void setExecutableName(std::string);
	void setFileNameDesignVector(std::string);
	void setID(unsigned int);
	unsigned int getID(void) const;

	bool checkFeasibility(double value) const;
	bool checkIfGradientAvailable(void) const;

	double calculateEI(rowvec x) const;

	void readEvaluateOutput(Design &d);
	void evaluate(Design &d);
	void evaluateAdjoint(Design &d);
	bool checkIfRunExecutableNecessary(void);

	double ftilde(rowvec x) const;
	void print(void) const;

	void addDesignToData(Design &d);
	std::string getExecutionCommand(void) const;
};




class ConstraintFunctionv2: public ObjectiveFunction {


private:
	double targetValue  = 0.0;
	std::string inequalityType;
	int ID = -1;
	bool ifInequalityConstraintSpecified  = false;

public:

	unsigned int readOutputStartIndex = 0;
	std::vector<int> IDToFunctionsShareOutputExecutable;

	ConstraintFunctionv2(std::string, unsigned int);
	ConstraintFunctionv2(std::string, double (*objFun)(double *), unsigned int);

	void readEvaluateOutput(Design &d);
	bool checkFeasibility(double value) const;




	void setInequalityConstraint(std::string inequalityStatement);

	void setID(int givenID) {
		assert(givenID>0);
		ID = givenID;

	}

	int getID(void) const {

		return ID;

	}

	void print(void) const;


};



#endif
