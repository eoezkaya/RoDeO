/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#ifndef CORRELATION_FUNCTIONS_HPP
#define CORRELATION_FUNCTIONS_HPP

#include <armadillo>
#include "correlation_functions.hpp"
using namespace arma;

class CorrelationFunction{

private:


	vec theta;             // hyper-parameter
	mat X;                 // input samples
	mat xtest;             // test samples
	mat correlationMatrix;
	mat correlationVec;

	bool ifInputSampleMatrixIsSet = false;

public:

	CorrelationFunction();

	void setInputSampleMatrix(mat);

	void corrgaussian_gekriging(mat &X, vec theta);
	mat corrbiquadspline_gekriging(mat &X,vec theta);

	void corrgaussian_gekriging_vec(mat &xtest, mat &X, vec theta);
	void corrbiquadspline_gekriging_vec(mat &xtest,mat &X, vec theta);

	void corrgaussian_kriging(mat &X,vec theta);
	void corrbiquadspline_kriging(mat &X,vec theta);

	void corrgaussian_kriging_vec(mat &xtest,mat &X,vec theta);
	void corrbiquadspline_kriging_vec(mat &xtest,mat &X,vec theta);

	bool isInputSampleMatrixSet(void) const;


};


class CorrelationFunctionBase{

protected:


	mat X;
	unsigned int N = 0;
	unsigned int dim = 0;
	mat correlationMatrix;
	mat correlationMatrixDot;
	vec correlationVec;

	double epsilon = 10E-012;

	bool ifInputSampleMatrixIsSet = false;


public:

	CorrelationFunctionBase();

	void setInputSampleMatrix(mat);
	void setEpsilon(double);
	void setDimension(unsigned int);
	virtual void setHyperParameters(vec) = 0;

	bool isInputSampleMatrixSet(void) const;

	mat getCorrelationMatrix(void) const;
	mat getCorrelationMatrixDot(void) const;

	void computeCorrelationMatrix(void);
	void computeCorrelationMatrixDot(void);

	mat compute_dCorrelationMatrixdxi(unsigned int k) const;
	mat compute_dCorrelationMatrixdxj(unsigned int k) const;
	mat compute_d2CorrelationMatrix_dxk_dxl(unsigned int k, unsigned l) const;


	virtual double compute_dR_dxi(const rowvec &, const rowvec &, unsigned int) const;
	virtual double compute_dR_dxj(const rowvec &, const rowvec &, unsigned int) const;
	virtual double compute_d2R_dxl_dxk(const rowvec &, const rowvec &, unsigned int ,unsigned int) const;

	vec computeCorrelationVector(const rowvec &x) const;

	virtual double computeCorrelation(const rowvec &x_i, const rowvec &x_j) const = 0;
	virtual bool checkIfParametersAreSetProperly(void) const = 0;


};

class ExponentialCorrelationFunction : public CorrelationFunctionBase{

private:


	vec theta;
	vec gamma;



public:


	void setTheta(vec);
	void setGamma(vec);

	void print(void) const;


	void initialize(void);
	void setHyperParameters(vec);
	vec getHyperParameters(void) const;
	double computeCorrelation(const rowvec &, const rowvec &) const;
	bool checkIfParametersAreSetProperly(void) const;



};


class BiQuadraticSplineCorrelationFunction : public CorrelationFunctionBase{

private:


	vec theta;


public:

	void setHyperParameters(vec);
	double computeCorrelation(const rowvec &, const rowvec &) const;
	bool checkIfParametersAreSetProperly(void) const;



};


class GaussianCorrelationFunctionForGEK : public CorrelationFunctionBase{

private:


	vec theta;


public:

	void initialize(void);

	void setHyperParameters(vec);
	vec getHyperParameters(void) const;

	double computeCorrelation(const rowvec &, const rowvec &) const;
	bool checkIfParametersAreSetProperly(void) const;

	double computeCorrelationDot(const rowvec &x_i, const rowvec &x_j, const rowvec &diffDirection) const;
	double computeCorrelationDotDot(const rowvec &x_i, const rowvec &x_j, const rowvec &firstDiffDirection, const rowvec &secondDiffDirection) const;
	double compute_dR_dxi(const rowvec &xi, const rowvec &xj, unsigned int k) const;
	double compute_dR_dxj(const rowvec &xi, const rowvec &xj, unsigned int k) const;
	double compute_d2R_dxl_dxk(const rowvec &, const rowvec &, unsigned int ,unsigned int) const;

	double computeCorrelation(unsigned int i, unsigned int j) const;
	double computeCorrelationDot(unsigned int i, unsigned int j, const rowvec &diffDirection) const;
	double computeCorrelationDotDot(unsigned int i, unsigned int j, const rowvec &firstDiffDirection, const rowvec &secondDiffDirection) const;



	void computeCorrelationMatrixDotForrester(void);


};




#endif
