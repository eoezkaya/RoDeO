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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#ifndef TGEK_HPP
#define TGEK_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "kriging_training.hpp"
#include "gaussian_correlation_function.hpp"
#include "linear_regression.hpp"
#include "linear_solver.hpp"
using namespace arma;



class TGEKModel : public SurrogateModel{

private:


	/* Auxiliary vectors */
	vec vectorOfOnes;

	vec w;
	mat weightMatrix;
	vec sampleWeights;

	mat Phi;
	vec ydot;

	SVDSystem  linearSystemCorrelationMatrixSVD;

	double beta0 = 0.0;
	double sigmaSquared = 0.0;

	double sigmaThresholdValueForSVD = 10E-12;

	bool ifUsesLinearRegression = false;
	bool ifCorrelationFunctionIsInitialized = false;
	bool ifActiveDeritiveSampleIndicesAreCalculated = false;


	double thetaFactor = 1.0;
	unsigned int numberOfIterationsToDetermineThetaFactor = 1000;
	vec theta;
	vec gamma;

	KrigingModel auxiliaryModel;

	ExponentialCorrelationFunction correlationFunction;
	GaussianCorrelationFunction differentiatedCorrelationFunction;


	double targetForDifferentiatedBasis = 0.0;
	vector<int> indicesDifferentiatedBasisFunctions;


	unsigned int numberOfDifferentiatedBasisFunctions = 0;

	vector<int> indicesOfSamplesWithActiveDerivatives;


	std::string filenameTrainingDataAuxModel = "auxiliaryData.csv";

	void calculatePhiEntriesForFunctionValues(void);
	void calculatePhiEntriesForDerivatives(void);
	void generateRhsForRBFs(void);
	void resetDataObjects(void);
	void updateModelWithNewData(void);
	void calculateBeta0(void);
	void setValuesForFindingDifferentiatedBasisIndex(vec &values);


public:

	bool ifVaryingSampleWeights = true;
	bool ifTargetForSampleWeightsIsSet = false;
	bool ifTargetForDifferentiatedBasisIsSet = false;

	void setName(string label);

	void setBoxConstraints(Bounds boxConstraintsInput);
	void setDimension(unsigned int);

	void readData(void);
	void normalizeData(void);
	void initializeCorrelationFunction(void);

	void setNameOfInputFile(string filename);
	void setNameOfHyperParametersFile(string filename);
	void setNumberOfTrainingIterations(unsigned int);
	void setNumberOfDifferentiatedBasisFunctionsUsed(unsigned int n);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);

	void setHyperParameters(vec parameters);

	void updateAuxilliaryFields(void);
	void train(void);
	double interpolate(rowvec x) const;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const ;

	void addNewSampleToData(rowvec newsample);
	void addNewLowFidelitySampleToData(rowvec newsample);


	void calculateIndicesOfSamplesWithActiveDerivatives(void);

	void findIndicesOfDifferentiatedBasisFunctionLocations(void);
	void initializeHyperParameters(void);

	void assembleLinearSystem(void);
	void solveLinearSystem();

	void calculatePhiMatrix(void);
	bool checkPhiMatrix(void);

	void trainTheta(void);
	void prepareTrainingDataForTheKrigingModel(void);
	void generateWeightingMatrix(void);
	void generateSampleWeights(void);

	mat getWeightMatrix(void) const;
	vec getSampleWeightsVector(void) const;
	vec  getHyperParameters(void) const;
	unsigned int getNumberOfSamplesWithActiveGradients(void) const;


};

#endif
