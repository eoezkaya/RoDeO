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

#include "standard_test_functions.hpp"
#include <armadillo>
#include<gtest/gtest.h>

using namespace arma;

TEST(testStandardTestFunctions, testconstructor){


	HimmelblauFunction testFun;

	unsigned int dim = testFun.getDimension();
	ASSERT_EQ(dim,2);


 }

TEST(testStandardTestFunctions, testevaluate){

	unsigned int dim = 2;
	HimmelblauFunction testFun;

	rowvec x(dim);
	x(0) = 3.0;
	x(1) = 2.0;

	double y = testFun.evaluate(x);
	ASSERT_EQ(y,0.0);
}


TEST(testStandardTestFunctions, testsetBoxConstraints){


	HimmelblauFunction testFun;
	testFun.setBoxConstraints(-6.0,6.0);

	Bounds boxConstraints = testFun.getBoxConstraints();

	double upperBound = boxConstraints.getUpperBound(0);
	ASSERT_EQ(upperBound,6.0);


}
