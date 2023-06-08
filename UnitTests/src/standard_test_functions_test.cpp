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

#ifdef STANDARD_TEST_FUNCTIONS_TEST


class EggholderTest : public ::testing::Test {
protected:
	void SetUp() override {}

	void TearDown() override {}


	EggholderFunction testFun;

};




TEST_F(EggholderTest, generateDataWithAdjoints){

	testFun.function.generateTrainingSamplesWithAdjoints();
	testFun.function.numberOfTestSamples = 200;
	testFun.function.generateTestSamples();

}


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

class Alpine02_5DTest : public ::testing::Test {
protected:
	void SetUp() override {}

	void TearDown() override {}


	Alpine02_5DFunction testFun;

};

TEST_F(Alpine02_5DTest, constructor){

	ASSERT_EQ(testFun.function.dimension,5);

}

TEST_F(Alpine02_5DTest, testTangent){

	Design d(5);
	testFun.function.evaluationSelect = 3;

	rowvec x(5);
	x(0) = 1.3; x(1) = 3.2; x(2) = 4.7; x(3) = 8.0; x(4) = 2.5;
	d.designParameters = x;

	rowvec dir(5);
	dir(0) = 1; dir(1) = 2; dir(2) = -1; dir(3) = 1; dir(4) = 0.0;

	d.tangentDirection = dir;

	testFun.function.evaluate(d);

	double tangentValue = d.tangentValue;
	double value = d.trueValue;

	double expectedValue = Alpine02_5D(x.memptr());

	double error = fabs(expectedValue - value);
	EXPECT_LT(error, 10E-10);

	double epsilon = 10E-6;
	rowvec xp = x + epsilon*dir;
	rowvec xm = x - epsilon*dir;

	double fp = Alpine02_5D(xp.memptr());
	double fm = Alpine02_5D(xm.memptr());

	double fdValue = (fp - fm)/(2.0*epsilon);

	error = fabs(fdValue - tangentValue);
	EXPECT_LT(error, 10E-6);

}


TEST_F(Alpine02_5DTest, testAdjoint){

	Design d(5);
	testFun.function.evaluationSelect = 2;

	rowvec x(5);
	x(0) = 1.3; x(1) = 3.2; x(2) = 4.7; x(3) = 8.0; x(4) = 2.5;
	d.designParameters = x;


	testFun.function.evaluate(d);

	rowvec gradient = d.gradient;


	rowvec dir(5,fill::zeros);
	dir(0) = 1.0;

	double epsilon = 10E-6;
	rowvec xp = x + epsilon*dir;
	rowvec xm = x - epsilon*dir;

	double fp = Alpine02_5D(xp.memptr());
	double fm = Alpine02_5D(xm.memptr());

	double fdValue = (fp - fm)/(2.0*epsilon);
	double gradValue = gradient(0);

	double error = fabs(fdValue - gradValue);
	EXPECT_LT(error, 10E-6);

	dir.fill(0.0);
	dir(1) = 1.0;
	xp = x + epsilon*dir;
	xm = x - epsilon*dir;

	fp = Alpine02_5D(xp.memptr());
	fm = Alpine02_5D(xm.memptr());

	fdValue = (fp - fm)/(2.0*epsilon);

	error = fabs(fdValue - gradient(1));
	EXPECT_LT(error, 10E-6);

	dir.fill(0.0);
	dir(0) = 0.5; dir(1) = -0.2;

	d.tangentDirection = dir;

	testFun.function.evaluationSelect = 3;
	testFun.function.evaluate(d);

	double tangentValue = d.tangentValue;
	double directionalDerivative = dot(dir,gradient);

	error = fabs(tangentValue - directionalDerivative);
	EXPECT_LT(error, 10E-10);

}


class WingweightFunctionTest : public ::testing::Test {
protected:
	void SetUp() override {}

	void TearDown() override {}


	WingweightFunction testFun;

};

TEST_F(WingweightFunctionTest, generateTrainingData){

	testFun.function.numberOfTrainingSamples = 50;
	testFun.function.generateTrainingSamples();


}

TEST_F(WingweightFunctionTest, testAdjoint){

	unsigned int dim = 10;

	Design d(dim);
	testFun.function.evaluationSelect = 2;

	rowvec x(dim);
	x(0) = 156.8; x(1) = 290.2; x(2) = 8.2; x(3) = -8.0; x(4) = 33.5;
	x(5) = 0.55; x(6) = 1.1; x(7) = 4.2; x(8) = 2000; x(9) = 0.03;
	d.designParameters = x;


	testFun.function.evaluate(d);

	rowvec gradient = d.gradient;

	double epsilon = 10E-6;
	for(unsigned int i=0; i<dim; i++){


		rowvec xp = x;
		xp(i) += epsilon;
		rowvec xm = x;
		xm(i) -= epsilon;

		double fp = Wingweight(xp.memptr());
		double fm = Wingweight(xm.memptr());
		double fdValue = (fp - fm)/(2.0*epsilon);
		double error = fabs(fdValue - gradient(i));
		EXPECT_LT(error, 10E-6);

	}

}

TEST_F(WingweightFunctionTest, evaluateGlobalExtrema){

	testFun.function.numberOfBruteForceIterations = 100000000;

	std::pair<double, double> extrema = testFun.function.evaluateGlobalExtrema();

	EXPECT_LT(extrema.first, 150.0);
	EXPECT_GT(extrema.second, 400.0);


}




#endif


