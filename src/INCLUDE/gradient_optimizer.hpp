/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#ifndef GRADIENTOPTIMIZER_HPP
#define GRADIENTOPTIMIZER_HPP


#include<armadillo>
#include "bounds.hpp"
#include "output.hpp"
#include "Rodeo_globals.hpp"
using namespace arma;

typedef double (*ObjectiveFunctionType)(vec);
typedef vec (*GradientFunctionType)(vec);


struct designPoint {
  vec x;
  vec gradient;
  double objectiveFunctionValue = 0;
  double L2NormGradient = 0;
} ;



class GradientOptimizer{

private:

	unsigned int dimension = 0;
	std::string optimizationType = "minimization";
	std::string LineSearchMethod = "backtracking_line_search";
	std::string finiteDifferenceMethod;
	bool ifOptimizationTypeMinimization = true;
	bool ifInitialPointIsSet = false;
	bool ifDimensionIsSet = false;
	bool ifObjectiveFunctionIsSet = false;
	bool ifGradientFunctionIsSet = false;
	bool areFiniteDifferenceApproximationToBeUsed = false;


	OutputDevice output;

	Bounds parameterBounds;

	designPoint currentIterate;
	designPoint nextIterate;


	std::vector<designPoint> designParameterHistory;

	double tolerance = 10E-6;
	double epsilonForFiniteDifferences = 0.001;
	double maximumStepSize = 0.0;
	unsigned int maximumNumberOfIterationsInLineSearch  = 10;

	double (*calculateObjectiveFunction)(vec);
	vec (*calculateGradientFunction)(vec);

	double optimalObjectiveFunctionValue = LARGE;

	unsigned int numberOfMaximumFunctionEvaluations = 0;
	unsigned int numberOfFunctionEvaluations = 0;

	virtual double calculateObjectiveFunctionInternal(vec);
	virtual vec calculateGradientFunctionInternal(vec);
	vec calculateCentralFiniteDifferences(designPoint &input);
	vec calculateForwardFiniteDifferences(designPoint &input);

public:

	GradientOptimizer();

	void setDimension(unsigned int dim);
	unsigned int getDimension(void) const;

	bool isOptimizationTypeMinimization(void) const;
	bool isOptimizationTypeMaximization(void) const;
	bool isInitialPointSet(void) const;
	bool isObjectiveFunctionSet(void) const;
	bool isGradientFunctionSet(void) const;

	bool areBoundsSet(void) const;

	void setDisplayOn(void);
	void setDisplayOff(void);
	void setNumberOfThreads(unsigned int nTreads);

	void setBounds(Bounds);

	void setInitialPoint(vec input);
	void setMaximumNumberOfFunctionEvaluations(unsigned int);
	void setObjectiveFunction(ObjectiveFunctionType functionToSet );
	void setGradientFunction(GradientFunctionType functionToSet );

	void setMaximumStepSize(double);
	void setTolerance(double);
	void setMaximumNumberOfIterationsInLineSearch(unsigned int);
	void setFiniteDifferenceMethod(string);
	void setEpsilonForFiniteDifference(double);

	void evaluateObjectiveFunction(designPoint &);
	void evaluateGradientFunction(designPoint &);
	void approximateGradientUsingFiniteDifferences(designPoint &input);

	void checkIfOptimizationSettingsAreOk(void) const;


	void performGoldenSectionSearch(void);
	void performBacktrackingLineSearch(void);
	void performLineSearch(void);
	void optimize(void);

	double getOptimalObjectiveFunctionValue(void) const;

};



#endif
