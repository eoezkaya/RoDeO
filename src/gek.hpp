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

#ifndef GEK_HPP
#define GEK_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "linear_regression.hpp"



using namespace arma;



class GEKModel : public SurrogateModel{

private:
	vec GEK_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_F;
	vec yGEK;
	vec vectorOfF;
	mat correlationMatrixDot;
	mat upperDiagonalMatrixDot;

	double beta0;
	double sigmaSquared;

	double epsilonGEK;

	double genErrorGEK;
	int maxNumberOfTrainingIterations;

	void updateWithNewData(void);
	void updateModelParams(void);


	double computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const;
	double computedR_dxj(rowvec x_i, rowvec x_j,int k) const;
	double computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const;
	double computedR_dxi(rowvec x_i, rowvec x_j,int k) const;

	void computeCorrelationMatrixDot(void);
	vec computeCorrelationVectorDot(rowvec x) const;

public:


	GEKModel();
	GEKModel(std::string name);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolateWithGradients(rowvec x) const ;
	double interpolate(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;


	double calculateExpectedImprovement(rowvec xp);
	double getyMin(void) const;
	vec getKrigingWeights(void) const;
	void setKrigingWeights(vec);
	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);
	void setNumberOfTrainingIterations(unsigned int);

	void resetDataObjects(void);
	void resizeDataObjects(void);
	int addNewSampleToData(rowvec newsample);
	void updateModelWithNewData(mat newData);
	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);

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
