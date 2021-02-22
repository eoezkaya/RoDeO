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
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"


class ConstraintFunction{


private:
	unsigned int ID;
	unsigned int dim;

	double (*pConstFun)(double *);
	double (*pConstFunAdj)(double *,double *);
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

	bool checkFeasibility(double value) const;
	bool checkIfGradientAvailable(void) const;

	double calculateEI(rowvec x) const;
//	double evaluate(rowvec x, bool);
	void evaluate(Design &d);
	void evaluateAdjoint(Design &d) const;
//	rowvec evaluateAdjoint(rowvec x,bool ifAddToData = true);
	double ftilde(rowvec x) const;
	void print(void) const;

	void addDesignToData(Design &d);
};



#endif
