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

class Correlationfunction{

private:


	vec theta;             // hyper-parameter
	mat X;                 // input samples
	mat xtest;             // test samples
	mat correlationMatrix;
	mat correlationVec;

public:

	Correlationfunction();

	void corrgaussian_gekriging(mat &X, vec theta);
	mat corrbiquadspline_gekriging(mat &X,vec theta);

	void corrgaussian_gekriging_vec(mat &xtest, mat &X, vec theta);
	void corrbiquadspline_gekriging_vec(mat &xtest,mat &X, vec theta);

	void corrgaussian_kriging(mat &X,vec theta);
	void corrbiquadspline_kriging(mat &X,vec theta);

	void corrgaussian_kriging_vec(mat &xtest,mat &X,vec theta);
	void corrbiquadspline_kriging_vec(mat &xtest,mat &X,vec theta);
};



#endif
