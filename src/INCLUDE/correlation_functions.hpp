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



#ifndef CORRELATION_FUNCTIONS_HPP
#define CORRELATION_FUNCTIONS_HPP

#include <armadillo>
#include "correlation_functions.hpp"
using namespace arma;



class CorrelationFunctionBase{

protected:


	mat X;
	unsigned int N = 0;
	unsigned int dim = 0;
	mat correlationMatrix;

	vec correlationVec;

	double epsilon = 10E-012;

	bool ifInputSampleMatrixIsSet = false;


public:

	CorrelationFunctionBase();

	void setInputSampleMatrix(mat);
	void setEpsilon(double);
	void setDimension(unsigned int);


	bool isInputSampleMatrixSet(void) const;

	mat getCorrelationMatrix(void) const;

	void computeCorrelationMatrix(void);


	vec computeCorrelationVector(const rowvec &x) const;

	virtual void setHyperParameters(vec) = 0;
	virtual double computeCorrelation(const rowvec &x_i, const rowvec &x_j) const = 0;

};



#endif
