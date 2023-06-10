/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
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
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */
#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "design.hpp"
#include "metric.hpp"


class AggregationModel : public SurrogateModel {

private:

	KrigingModel krigingModel;

	double rho = 1.0;

	unsigned int numberOfTrainingIterations;

	WeightedL1Norm weightedL1norm;

	string filenameTrainingForKrigingModel = "AggModel_primalTrainingData.csv";




public:


	void setDimension(unsigned int dim);

	void setBoxConstraints(Bounds boxConstraintsInput);

	void readData(void);
	void normalizeData(void);

	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);
	void setNumberOfTrainingIterations(unsigned int);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void updateAuxilliaryFields(void);
	void prepareTrainingAndTestData(void);
	void train(void);
	void determineRhoBasedOnData(void);
	void determineOptimalL1NormWeights(void);
	void setRho(double);

	vec getL1NormWeights(void) const;

	double calculateMinimumDistanceToNearestPoint(const rowvec &, int) const;
	double calculateDualModelEstimate(const rowvec &, int) const;
	double calculateDualModelWeight(const rowvec &, int) const;

	double interpolate(rowvec) const ;
	double interpolateWithGradients(rowvec) const ;
	void interpolateWithVariance(rowvec,double *,double *) const;

	unsigned int findNearestNeighbor(const rowvec &) const;
	void addNewSampleToData(rowvec);
	void updateModelWithNewData(void);
	void addNewLowFidelitySampleToData(rowvec newsample);


};



#endif
