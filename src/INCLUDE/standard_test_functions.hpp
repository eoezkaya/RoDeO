/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#ifndef STANDARD_TEST_FUNCTIONS_HPP
#define STANDARD_TEST_FUNCTIONS_HPP

#include <armadillo>
#include "test_functions.hpp"


class HimmelblauFunction{


public:

	TestFunction function;

	HimmelblauFunction();
	HimmelblauFunction(double, double);



};


class LinearTestFunction1{


public:

	TestFunction function;

	LinearTestFunction1();
	LinearTestFunction1(double, double);



};


class NonLinear1DTestFunction1{


public:

	TestFunction function;

	NonLinear1DTestFunction1();
	NonLinear1DTestFunction1(double, double);



};



#endif
