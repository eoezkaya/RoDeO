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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "standard_test_functions.hpp"
#include "test_defines.hpp"
#include <armadillo>
#include<gtest/gtest.h>

using namespace arma;

#ifdef TEST_STANDARD_TEST_FUNCTIONS


class HimmelblauTest : public ::testing::Test {
protected:
	void SetUp() override {}

	void TearDown() override {}


	HimmelblauFunction testFun;

};


TEST_F(HimmelblauTest, constructor){

	ASSERT_EQ(testFun.function.dimension,2);

}

TEST_F(HimmelblauTest, testTangent){

	Design d(2);
	testFun.function.evaluationSelect = 3;

	rowvec x(2);
	x(0) = 1.3; x(1) = -3.2;
	d.designParameters = x;

	rowvec dir(2);
	dir(0) = 0.5; dir(1) = -0.2;

	d.tangentDirection = dir;

	testFun.function.evaluate(d);

	double tangentValue = d.tangentValue;
	double value = d.trueValue;

	double expectedValue = Himmelblau(x.memptr());

	double error = fabs(expectedValue - value);
	EXPECT_LT(error, 10E-10);

	double epsilon = 10E-6;
	rowvec xp = x + epsilon*dir;
	rowvec xm = x - epsilon*dir;

	double fp = Himmelblau(xp.memptr());
	double fm = Himmelblau(xm.memptr());

	double fdValue = (fp - fm)/(2.0*epsilon);

	error = fabs(fdValue - tangentValue);
	EXPECT_LT(error, 10E-6);

}


TEST_F(HimmelblauTest, testAdjoint){

	Design d(2);
	testFun.function.evaluationSelect = 2;

	rowvec x(2);
	x(0) = 1.3; x(1) = -3.2;
	d.designParameters = x;


	testFun.function.evaluate(d);

	rowvec gradient = d.gradient;


	rowvec dir(2);
	dir(0) = 1; dir(1) = 0;

	double epsilon = 10E-6;
	rowvec xp = x + epsilon*dir;
	rowvec xm = x - epsilon*dir;

	double fp = Himmelblau(xp.memptr());
	double fm = Himmelblau(xm.memptr());

	double fdValue = (fp - fm)/(2.0*epsilon);

	double error = fabs(fdValue - gradient(0));
	EXPECT_LT(error, 10E-6);

	dir(0) = 0; dir(1) = 1;
	xp = x + epsilon*dir;
	xm = x - epsilon*dir;

	fp = Himmelblau(xp.memptr());
	fm = Himmelblau(xm.memptr());

	fdValue = (fp - fm)/(2.0*epsilon);

	error = fabs(fdValue - gradient(1));
	EXPECT_LT(error, 10E-6);


	dir(0) = 0.5; dir(1) = -0.2;

	d.tangentDirection = dir;

	testFun.function.evaluationSelect = 3;
	testFun.function.evaluate(d);

	double tangentValue = d.tangentValue;
	double directionalDerivative = dot(dir,gradient);

	error = fabs(tangentValue - directionalDerivative);
	EXPECT_LT(error, 10E-10);

}



#endif


