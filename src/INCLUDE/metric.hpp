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

#ifndef METRIC
#define METRIC
#include <armadillo>
using namespace arma;

class WeightedL1Norm{

private:
	unsigned int dim = 0;
	vec weights;
	mat trainingData;
	mat validationData;

	unsigned int nTrainingIterations = 1000;


public:

	bool ifDisplay = false;


	WeightedL1Norm();
	WeightedL1Norm(unsigned int);
	WeightedL1Norm(vec);

	void setTrainingData(mat);
	void setValidationData(mat);
	void setNumberOfTrainingIterations(unsigned int);

	double calculateNorm(const rowvec &) const;
	void findOptimalWeights();
	double interpolateByNearestNeighbour(rowvec) const;
	double calculateMeanSquaredErrorOnData(void) const;
	void generateRandomWeights(void);


};



double calculateL1norm(const rowvec &x);
double calculateWeightedL1norm(const rowvec &x, vec w);
double calculateMetric(rowvec &xi,rowvec &xj, mat M);
double calculateMetricAdjoint(rowvec xi, rowvec xj, mat M, mat &Mb, double calculateMetricb);
unsigned int findNearestNeighborL1(const rowvec &xp, const mat &X);


#endif
