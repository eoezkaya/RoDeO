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

#ifndef TGEK_HPP
#define TGEK_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "kriging_training.hpp"
#include "correlation_functions.hpp"
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
	   	mat WPhi;
	   	vec ydot;
	   	vec Wydot;

	   	SVDSystem  linearSystemCorrelationMatrixSVD;

	   	double beta0 = 0.0;
	   	double sigmaSquared = 0.0;

	   	bool ifUsesLinearRegression = false;
	   	bool ifCorrelationFunctionIsInitialized = false;

	   	double genError;

	   	LinearModel linearModel;
	   	KrigingModel auxiliaryModel;

	   	GaussianCorrelationFunctionForGEK correlationFunction;


	   	uvec indicesDifferentiatedBasisFunctions;

	   	std::string filenameTrainingDataAuxModel = "auxiliaryData.csv";

	void calculatePhiEntriesForFunctionValues(void);
	void calculatePhiEntriesForDerivatives(void);
	void generateRhsForRBFs(void);
	void resetDataObjects(void);
	void updateModelWithNewData(void);


public:

	   	unsigned int numberOfDifferentiatedBasisFunctions = 5;

	   	void readData(void);

		void setNameOfInputFile(string filename);
		void setNameOfHyperParametersFile(string filename);
		void setNumberOfTrainingIterations(unsigned int);
		void setNumberOfDifferentiatedBasisFunctionsUsed(unsigned int n);

		void initializeSurrogateModel(void);
		void printSurrogateModel(void) const;
		void printHyperParameters(void) const;
		void saveHyperParameters(void) const;
		void loadHyperParameters(void);
		vec  getHyperParameters(void) const;
		void setHyperParameters(vec parameters);

		void updateAuxilliaryFields(void);
		void train(void);
		double interpolate(rowvec x) const;
		void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const ;
//		void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;
		void addNewSampleToData(rowvec newsample);
		void addNewLowFidelitySampleToData(rowvec newsample);



		void findIndicesOfDifferentiatedBasisFunctionLocations(void);
		void initializeHyperParameters(void);
		void calculatePhiMatrix(void);
		bool checkPhiMatrix(void);

		void trainTheta(void);
		void prepareTrainingDataForTheKrigingModel(void);
		void generateWeightingMatrix(void);
		void generateSampleWeights(void);

		mat getWeightMatrix(void) const;
		vec getSampleWeightsVector(void) const;




};

#endif
