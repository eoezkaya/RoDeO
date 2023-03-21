/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), Rheinland-Pfälzische Technische Universität
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
#include "Rodeo_globals.hpp"
using namespace arma;

class DesignForBayesianOptimization{

public:
	rowvec dv;
	double objectiveFunctionValue = 0.0;
	double valueAcqusitionFunction = 0.0;

	double sigma = 0.0;
	rowvec constraintValues;
	rowvec constraintSigmas;
	rowvec constraintFeasibilityProbabilities;
	unsigned int dim = 0;


	DesignForBayesianOptimization();
	DesignForBayesianOptimization(unsigned int dimension, unsigned int numberOfConstraints);
	DesignForBayesianOptimization(unsigned int dimension);
	DesignForBayesianOptimization(rowvec designVector, unsigned int numberOfConstraints);
	DesignForBayesianOptimization(rowvec designVector);

	void generateRandomDesignVector(void);
	void generateRandomDesignVector(vec lb, vec ub);

	void generateRandomDesignVectorAroundASample(const rowvec &sample, vec lb, vec ub);


	double calculateProbalityThatTheEstimateIsLessThanAValue(double value);
	double calculateProbalityThatTheEstimateIsGreaterThanAValue(double value);

	void updateAcqusitionFunctionAccordingToConstraints(void);


	void gradientUpdateDesignVector(const rowvec &, const vec &, const vec &, double);
	void print(void) const;


};




class Design{

public:

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;

	rowvec designParameters;
	rowvec designParametersNormalized;
	rowvec constraintTrueValues;
	rowvec constraintEstimates;
	rowvec gradient;
	rowvec gradientLowFidelity;
	rowvec tangentDirection;

	double trueValue = 0.0;
	double estimatedValue = 0.0;
	double trueValueLowFidelity = 0;
	double tangentValue = 0.0;
	double tangentValueLowFidelity = 0.0;
	double improvementValue = 0.0;

	double ExpectedImprovementvalue = 0.0;

	std::vector<rowvec> constraintGradients;
	std::vector<rowvec> constraintGradientsLowFidelity;
	std::vector<rowvec> constraintTangentDirection;


	mat constraintGradientsMatrix;
	mat constraintGradientsMatrixLowFi;
	mat constraintDifferentiationDirectionsMatrix;
	mat constraintDifferentiationDirectionsMatrixLowFi;




	rowvec constraintTangent;
	rowvec constraintTangentLowFidelity;
	rowvec constraintTrueValuesLowFidelity;

	double surrogateEstimate = 0.0;
	rowvec constraintSurrogateEstimates;

	unsigned int ID = 0;
	std::string tag;

	bool isDesignFeasible = true;

	Design();
	Design(rowvec);
	Design(unsigned int);

	void setNumberOfConstraints(unsigned int howManyConstraints);

	void print(void) const;

	void saveToAFile(std::string) const;
	void saveDesignVectorAsCSVFile(std::string fileName) const;
	void saveDesignVector(std::string fileName) const;

	void generateRandomDesignVector(vec lb, vec ub);
	void generateRandomDesignVector(double lb, double ub);

	void setDimension(unsigned int);


	bool checkIfHasNan(void) const;

	void calculateNormalizedDesignParameters(vec lb, vec ub);
	void calculateDesignParametersFromNormalized(vec lb, vec ub);


	rowvec constructSampleObjectiveFunction(void) const;
	rowvec constructSampleObjectiveFunctionLowFi(void) const;
	rowvec constructSampleObjectiveFunctionWithGradient(void) const;
	rowvec constructSampleObjectiveFunctionWithGradientLowFi(void) const;
	rowvec constructSampleObjectiveFunctionWithTangent(void) const;
	rowvec constructSampleObjectiveFunctionWithTangentLowFi(void) const;

	rowvec constructSampleConstraint(int constraintID) const;
	rowvec constructSampleConstraintLowFi(int constraintID) const;
	rowvec constructSampleConstraintWithTangent(int constraintID) const;
	rowvec constructSampleConstraintWithTangentLowFi(int constraintID) const;
	rowvec constructSampleConstraintWithGradient(int constraintID) const;
	rowvec constructSampleConstraintWithGradientLowFi(int constraintID) const;


};




#endif
