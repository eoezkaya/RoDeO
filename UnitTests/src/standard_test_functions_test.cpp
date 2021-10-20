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

TEST(testStandardTestFunctions, testevaluateGradient){

	unsigned int dim = 2;
	HimmelblauFunction testFun;

	rowvec x(dim);
	x(0) = 3.0;
	x(1) = 2.0;

	rowvec gradient(dim);

	testFun.evaluateGradient(x, gradient);

	double error0 = fabs(0.0 - gradient(0));
	ASSERT_LT(error0,10E-10);
	double error1 = fabs(0.0 - gradient(1));
	ASSERT_LT(error1,10E-10);
}



TEST(testStandardTestFunctions, testsetBoxConstraints){


	HimmelblauFunction testFun;
	testFun.setBoxConstraints(-6.0,6.0);

	Bounds boxConstraints = testFun.getBoxConstraints();

	double upperBound = boxConstraints.getUpperBound(0);
	ASSERT_EQ(upperBound,6.0);


}

TEST(testStandardTestFunctions, testgenerateSamples){

	unsigned int N = 100;
	unsigned int dim = 2;
	HimmelblauFunction testFun;
	testFun.setBoxConstraints(-6.0,6.0);

	testFun.generateSamples(N);

	mat samples = testFun.getSamples();

	unsigned int nRows = samples.n_rows;
	unsigned int nCols = samples.n_cols;

	ASSERT_EQ(nRows,100);
	ASSERT_EQ(nCols,dim+1);

	rowvec x = samples.row(17);

	double functionValue = pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );

	double error = fabs( x(dim) - functionValue );

	ASSERT_LT(error, 10E-10);


}

TEST(testStandardTestFunctions, testgenerateSamplesWithGradient){

	unsigned int N = 100;
	unsigned int dim = 2;
	HimmelblauFunction testFun;
	testFun.setBoxConstraints(-6.0,6.0);

	testFun.generateSamplesWithGradient(N);

	mat samples = testFun.getSamples();

	unsigned int nRows = samples.n_rows;
	unsigned int nCols = samples.n_cols;

	ASSERT_EQ(nRows,100);
	ASSERT_EQ(nCols,2*dim+1);

}
