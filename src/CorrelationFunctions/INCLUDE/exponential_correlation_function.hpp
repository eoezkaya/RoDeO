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



#ifndef EXP_CORRELATION_FUNCTION_HPP
#define EXP_CORRELATION_FUNCTION_HPP

#include <armadillo>
#include "correlation_functions.hpp"
using namespace arma;


class ExponentialCorrelationFunction : public CorrelationFunctionBase{

private:


	vec thetaParameters;
	vec gammaParameters;
	double alpha = 1.0;



public:


	void setTheta(vec);
	void setGamma(vec);
	void setAlpha(double);

	void print(void) const;


	void initialize(void);
	void setHyperParameters(vec);
	vec getHyperParameters(void) const;
	double computeCorrelation(const rowvec &, const rowvec &) const;
	double computeCorrelationDot(const rowvec &, const rowvec &, const rowvec &) const;

	bool checkIfParametersAreSetProperly(void) const;



};

#endif
