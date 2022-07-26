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

#ifndef METRIC
#define METRIC
#include <armadillo>
#include "ea_optimizer.hpp"
#include "gradient_optimizer.hpp"
using namespace arma;

class WeightedL1Norm{

private:
	unsigned int dimension = 0;
	vec weights;
	mat trainingData;
	mat inputTrainingData;
	mat validationData;
	mat inputValidationData;
	vec outputTrainingData;
	vec outputValidationData;

	unsigned int nTrainingIterations = 0;
	unsigned int numberOfThreads = 1;
	bool ifTrainingDataIsSet = false;
	bool ifValidationDataIsSet = false;
	bool ifNumberOfTrainingIterationsIsSet = false;

	void initializeNumberOfTrainingIterations();


	OutputDevice output;

public:

	bool ifDisplay = false;



	WeightedL1Norm();
	WeightedL1Norm(unsigned int);
	WeightedL1Norm(vec);

	void initialize(unsigned int);

	void setTrainingData(mat);
	void setValidationData(mat);
	void setNumberOfTrainingIterations(unsigned int);
	void setNumberOfThreads(unsigned int);

	bool isTrainingDataSet(void) const;
	bool isValidationDataSet(void) const;
	bool isNumberOfTrainingIterationsSet(void) const;

	void setDimension(unsigned int dim);
	unsigned int getDimension(void) const;
	vec getWeights(void) const;
	void setWeights(vec wIn);

	double calculateNorm(const rowvec &) const;
	void findOptimalWeights();
	void findOptimalWeightsGradientBased(vec);

	double interpolateByNearestNeighbor(rowvec) const;
	int findNearestNeighbor(const arma::rowvec &x) const;
	double calculateMeanSquaredErrorOnData(void) const;
	double calculateMeanL1ErrorOnData(void) const;
	void generateRandomWeights(void);



};


class WeightedL1NormOptimizer : public EAOptimizer{

private:

	double calculateObjectiveFunctionInternal(vec& input);
	WeightedL1Norm  weightedL1NormForCalculations;

	bool ifWeightedL1NormForCalculationsIsSet = false;


public:

	bool isWeightedL1NormForCalculationsSet(void) const;

	void initializeWeightedL1NormObject(WeightedL1Norm);


};




double calculateL1norm(const rowvec &x);
double calculateWeightedL1norm(const rowvec &x, vec w);
double calculateMetric(rowvec &xi,rowvec &xj, mat M);
unsigned int findNearestNeighborL1(const rowvec &xp, const mat &X);


#endif
