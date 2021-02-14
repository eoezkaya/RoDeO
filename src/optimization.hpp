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
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP


#include <armadillo>
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"






class ObjectiveFunction{


private:

	double (*objectiveFunPtr)(double *);
	std::string executableName;
	std::string executablePath;
	std::string fileNameObjectiveFunctionRead;
	std::string fileNameDesignVector;

	KrigingModel surrogateModel;
	AggregationModel surrogateModelGradient;
	unsigned int dim;
	bool ifDoErequired;
	bool ifWarmStart;
	bool ifGradientAvailable;

public:

	std::string name;
	ObjectiveFunction(std::string, double (*objFun)(double *), unsigned int);
	ObjectiveFunction(std::string, unsigned int);
	ObjectiveFunction();

	void readConfigFile(void);

	void trainSurrogate(void);

	void saveDoEData(mat) const;
	void setFileNameReadObjectFunction(std::string);
	void setExecutablePath(std::string);
	void setExecutableName(std::string);
	void setFileNameDesignVector(std::string);
	double calculateExpectedImprovement(rowvec x);
	double evaluate(rowvec x, bool);
	double ftilde(rowvec x) const;
	void print(void) const;

};




class ConstraintFunction{


private:
	unsigned int ID;
	unsigned int dim;

	double (*pConstFun)(double *);
	double targetValue;
	std::string inequalityType;

	KrigingModel surrogateModel;
	AggregationModel surrogateModelGradient;

	std::string executableName;
	std::string executablePath;
	std::string fileNameConstraintFunctionRead;
	std::string fileNameDesignVector;

	bool ifGradientAvailable;

public:
	std::string name;
	bool ifNeedsSurrogate;
	std::vector<int> IDToFunctionsShareOutputFile;
	std::vector<int> IDToFunctionsShareOutputExecutable;


	ConstraintFunction(std::string, std::string, double, double (*constFun)(double *), unsigned int dimension, bool ifNeedsSurrogate = false);
	ConstraintFunction(std::string, std::string, double, unsigned int);
	ConstraintFunction();
	void saveDoEData(mat) const;
	void trainSurrogate(void);

	void setFileNameReadConstraintFunction(std::string);
	void setExecutablePath(std::string);
	void setExecutableName(std::string);
	void setFileNameDesignVector(std::string);
	void setID(int);

	bool checkFeasibility(double value);

	double calculateEI(rowvec x) const;
	double evaluate(rowvec x, bool);
	double ftilde(rowvec x) const;
	void print(void) const;
};




class Optimizer {

private:

	vec lowerBounds;
	vec upperBounds;
	mat optimizationHistory;


	std::vector<ConstraintFunction> constraintFunctions;
	ObjectiveFunction objFun;



public:

	std::string name;
	unsigned int dimension;
	unsigned int numberOfConstraints;
	unsigned int maxNumberOfSamples;
	unsigned int howOftenTrainModels;

	unsigned int iterGradientEILoop;
	std::string optimizationType;
	bool ifVisualize;

	double epsilon_EI;
	unsigned int iterMaxEILoop;

	bool ifBoxConstraintsSet;

	Optimizer(std::string ,int, std::string);
	void print(void) const;
	void printConstraints(void) const;
	void visualizeOptimizationHistory(void) const;
	void EfficientGlobalOptimization(void);
	void trainSurrogates(void);
	void performDoE(unsigned int howManySamples, DoE_METHOD methodID);

	void setProblemType(std::string);
	void setMaximumNumberOfIterations(unsigned int );
	void setBoxConstraints(std::string filename="BoxConstraints.csv");
	void setBoxConstraints(double lb, double ub);
	void setBoxConstraints(vec lb, vec ub);

	void addConstraint(ConstraintFunction &constFunc);

	void evaluateConstraints(rowvec x, rowvec &constraintValues,bool);
	void estimateConstraints(rowvec x, rowvec &constraintValues);

	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec constraintValues);

	void addObjectFunction(ObjectiveFunction &objFunc);


};

void testOptimizationHimmelblau(void);
void testOptimizationHimmelblauExternalExe(void);
void testOptimizationWingweight(void);
void testOptimizationEggholder(void);

#endif
