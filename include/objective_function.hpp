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
#ifndef OBJECTIVE_FUNCTION_HPP
#define OBJECTIVE_FUNCTION_HPP


#include <armadillo>
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "design.hpp"


class ObjectiveFunction{


private:



protected:

	std::string name;
	double (*objectiveFunPtr)(double *);
	double (*objectiveFunAdjPtr)(double *,double *);
	std::string executableName;
	std::string executablePath;
	std::string fileNameInputRead;
	std::string fileNameDesignVector;

	vec ub;
	vec lb;

	KrigingModel surrogateModel;
	AggregationModel surrogateModelGradient;

	unsigned int dim = 0;
	bool ifDoErequired = true;
	bool ifWarmStart = false;
	bool ifGradientAvailable = false;
	bool ifFunctionPointerIsSet = false;
	bool ifInitialized = false;
	bool ifParameterBoundsAreSet = false;

public:


	ObjectiveFunction(std::string, double (*objFun)(double *), unsigned int);
	ObjectiveFunction(std::string, double (*objFun)(double *, double *), unsigned int);
	ObjectiveFunction(std::string, unsigned int);
	ObjectiveFunction();


	void initializeSurrogate(void);
	void trainSurrogate(void);
	KrigingModel getSurrogateModel(void) const;
	AggregationModel getSurrogateModelGradient(void) const;


	void setGradientOn(void);
	void setGradientOff(void);
	void setParameterBounds(vec , vec );

	unsigned int getDimension(void) const{

		return dim;
	}

	std::string getName(void) const{

		return name;
	}

	bool ifHasFunctionFunctionPointer(void) const{

		return ifFunctionPointerIsSet;

	}


	void setFileNameReadInput(std::string fileName){

		fileNameInputRead = fileName;

	}


	void saveDoEData(std::vector<rowvec>) const;
	void setExecutablePath(std::string);
	void setExecutableName(std::string);
	void setFileNameDesignVector(std::string);
	double calculateExpectedImprovement(rowvec x) const;

	void evaluate(Design &d);
	void evaluateAdjoint(Design &d);
	void readEvaluateOutput(Design &d);


	void addDesignToData(Design &d);

	bool checkIfGradientAvailable(void) const;
	double ftilde(rowvec x, bool ifdebug = false) const;
	void print(void) const;
	std::string getExecutionCommand(void) const;
};


#endif
