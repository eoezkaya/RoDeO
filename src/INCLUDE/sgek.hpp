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

#ifndef SGEK_HPP
#define SGEK_HPP
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "linear_regression.hpp"
#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
using namespace arma;

class SGEKModel : public SurrogateModel{

private:

	vec GEK_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_F;
	vec yGEK;
	vec vectorOfF;
	mat correlationMatrixDot;
	mat upperDiagonalMatrixDot;
	vec sensitivity;    // derivative-based global sensitivity analysis
	mat gradient;       // gradient matrix
	vec alpha;          // auxiliary hyper-parameter

	unsigned int snum;  // slice number
	double threshold;   // truncation threshold
	unsigned int edim;  // effective dimension

	double likelihood;
	double beta0;
	double sigmaSquared;

	double epsilonGEK;

	double genErrorGEK;
	int maxNumberOfTrainingIterations;
	field<uvec> index;

	void updateWithNewData(void);
	void updateModelParams(void);

	Correlationfunction correlationfunction;

	double computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const;
	double computedR_dxj(rowvec x_i, rowvec x_j,int k) const;
	double computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const;
	double computedR_dxi(rowvec x_i, rowvec x_j,int k) const;
	double sliced_likelihood_function(vec alpha);         // Modified by Kai
	void original_likelihood_function(vec theta);  // Modified by Kai
	void slicing(unsigned int snum);
	void computeCorrelationMatrixDot(vec theta);
	vec computeCorrelationVectorDot(rowvec x) const;

	/* Added by Kai

	field<mat>  X_2;
	field<mat>  upperDiagonalMatrixDot_2;
	field<mat>  correlationMatrixDot_2;

	field<vec>  yGEK_2;
	field<vec>  R_inv_F_2;
	field<vec>  vectorOfF_2;
	field<vec>  R_inv_ys_2;
	field<vec>  R_inv_ys_min_beta_2;
	field<vec>  ys_min_betaF_2;
	field<uvec> index_2;

	vec logdetR_2;
	vec n_2;
	vec nom_2;
	vec denom_2;
	vec sig_2;


	field<mat>  X_1;
	field<mat>  upperDiagonalMatrixDot_1;
	field<mat>  correlationMatrixDot_1;

	field<vec>  yGEK_1;
	field<vec>  R_inv_F_1;
	field<vec>  vectorOfF_1;
	field<vec>  R_inv_ys_1;
	field<vec>  R_inv_ys_min_beta_1;
	field<vec>  ys_min_betaF_1;
	field<uvec> index_1;

	vec logdetR_1;
	vec n_1;
	vec nom_1;
	vec denom_1;
	vec sig_1; */

	/* Added by Kai */

    int num;    // number of multiple starts
	vec hyper_lb;
	vec hyper_up;
	vec hyper_in;

    vec hyper_cur;
	vec hyper_par;

	vec increment;
	uvec ind_increment;

	mat hyperoptimizationHistory;

	unsigned int numberOfIteration;

	int dim;
	int dim_a;

	double likelihood_cur;

public:


	SGEKModel();
	SGEKModel(std::string name);

	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);
	void setNumberOfTrainingIterations(unsigned int);


	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolateWithGradients(rowvec x) const ;
	double interpolate(rowvec x) const ;

	mat interpolate_all(mat x);

	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;
	void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;
	void addNewSampleToData(rowvec newsample);


	double calculateExpectedImprovement(rowvec xp);
	double getyMin(void) const;
	vec getKrigingWeights(void) const;
	void setKrigingWeights(vec);
	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);

	void resetDataObjects(void);
	void resizeDataObjects(void);
	void updateModelWithNewData(mat newData);
	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);

	/* Hooke Jeeves algorithm */

	void boxmin(vec hyper_lb, vec hyper_ub, int num);
    void start(vec int_hyper, vec hyper_lb, vec hyper_ub);
	void explore(vec int_hyper, double likelihood);
    void move(vec hyper_1, vec hyper_2, double likelihood);
    vec getTheta(void) const;
    vec getAlpha(void) const;
    double getLikelihood(void) const;
	/* test functions */

	friend void testGEKcalculateRDot(void);
	friend void testGEKcalculateRDotValidateWithWingweight(void);
	friend void testGEKcalculateCorrelationVectorDotWithWingweight(void);
	friend void testGEKWithWingweight(void);
	friend void testGEKValueOfMuWithWingweight(void);
	friend void testGEKPredictionWithWingweight(void);
	friend void testGEKPredictionWithWaves(void);

};

#endif
