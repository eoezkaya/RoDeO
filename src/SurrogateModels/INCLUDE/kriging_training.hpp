/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP




#include <armadillo>
#include "../../INCLUDE/Rodeo_macros.hpp"
#include "./surrogate_model.hpp"
#include "./linear_regression.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"
#include "../../LinearAlgebra/INCLUDE/linear_solver.hpp"
#include "../../Optimizers/INCLUDE/ea_optimizer.hpp"
#include "../../CorrelationFunctions/INCLUDE/exponential_correlation_function.hpp"
using namespace arma;



class KrigingModel : public SurrogateModel{

private:

	/* Auxiliary vectors */
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	vec R_inv_ys;
	vec vectorOfOnes;


	CholeskySystem linearSystemCorrelationMatrix;
	SVDSystem      linearSystemCorrelationMatrixSVD;

	double beta0 = 0.0;
	double sigmaSquared = 0.0;

	bool ifUsesLinearRegression = false;
	bool ifCorrelationFunctionIsInitialized = false;

	double genErrorKriging;

	LinearModel linearModel;

	ExponentialCorrelationFunction correlationFunction;

	void updateWithNewData(void);
	void updateModelParams(void);
	void calculateBeta0(void);


public:

	KrigingModel();


	void readData(void);
	void normalizeData(void);

	void setBoxConstraints(Bounds boxConstraintsInput);

	void setDimension(unsigned int);
	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);


	void setDisplayOn(void);
	void setDisplayOff(void);




	void setNumberOfTrainingIterations(unsigned int);


	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void setHyperParameters(vec);
	vec getHyperParameters(void) const;

	void train(void);

	double interpolate(rowvec x) const ;
	double interpolateUsingDerivatives(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;


	void addNewSampleToData(rowvec newsample);
	void addNewLowFidelitySampleToData(rowvec newsample);


	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);

	mat getCorrelationMatrix(void) const;

	void resetDataObjects(void);
	void resizeDataObjects(void);


	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);
	void updateAuxilliaryFieldsWithSVDMethod(void);
	void checkAuxilliaryFields(void) const;

	double calculateLikelihoodFunction(vec);


};


class KrigingHyperParameterOptimizer : public EAOptimizer{

private:

	double calculateObjectiveFunctionInternal(vec& input);
	KrigingModel  KrigingModelForCalculations;



public:

	void initializeKrigingModelObject(KrigingModel);
	bool ifModelObjectIsSet = false;



};






#endif
