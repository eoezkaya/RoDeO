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
#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "design.hpp"


class AggregationModel : public SurrogateModel {

private:

	KrigingModel krigingModel;

	vec L1NormWeights;

	double rho;


	PartitionData trainingDataForHyperParameterOptimization;
	PartitionData testDataForHyperParameterOptimization;

	unsigned int numberOfTrainingIterations;


public:


	AggregationModel();
	AggregationModel(std::string name);

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
	void setNumberOfTrainingIterations(unsigned int numberOfIterations){

		this->numberOfTrainingIterations = numberOfIterations;
	}

	vec getL1NormWeights(void) const;
	PartitionData getTrainingData(void) const;
	PartitionData getTestData(void) const;

	void generateRandomHyperParams(void);
	double interpolate(rowvec x, bool ifprint = false) const ;
	double interpolateWithGradients(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

//	double calculateExpectedImprovement(rowvec xp) const;
	void calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const;


	unsigned int findNearestNeighbor(const rowvec &) const;
	void addNewSampleToData(rowvec newsample);
	void updateModelWithNewData(void);
	void modifyRawDataAndAssociatedVariables(mat dataMatrix);


};



#endif
