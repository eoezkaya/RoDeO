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

#include "exponential_correlation_function.hpp"
#include "auxiliary_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef EXP_CORRELATION_FUNCTIONS_TEST

class ExpCorrelationFunctionTest : public ::testing::Test {
protected:
	void SetUp() override {


		testCorrelationFunction.setDimension(3);

	}

	void TearDown() override {}
	ExponentialCorrelationFunction testCorrelationFunction;
};


TEST_F(ExpCorrelationFunctionTest, testinitialize){


	ASSERT_FALSE(testCorrelationFunction.isInputSampleMatrixSet());

}

TEST_F(ExpCorrelationFunctionTest, testSetInputSampleMatrix){

	unsigned int N = 100;
	unsigned dim = 5;
	mat testInput(N,dim,fill::randu);

	testCorrelationFunction.setInputSampleMatrix(testInput);
	ASSERT_TRUE(testCorrelationFunction.isInputSampleMatrixSet());

}


TEST_F(ExpCorrelationFunctionTest, testComputeCorrelation){

	testCorrelationFunction.initialize();
	rowvec x1(3,fill::randu);
	rowvec x2(3,fill::randu);

	double R = testCorrelationFunction.computeCorrelation(x1,x2);

	double Rexpected = exp(- ((x1(0)-x2(0))*(x1(0)-x2(0)) + (x1(1)-x2(1))*(x1(1)-x2(1)) + (x1(2)-x2(2))*(x1(2)-x2(2)) ));
	double error = fabs(Rexpected - R);
	EXPECT_LT(error,10E-10);

}

TEST_F(ExpCorrelationFunctionTest, testComputeCorrelationDot){

	testCorrelationFunction.initialize();


	const double eps = 10E-010;

	for(unsigned int i=0; i<1000; i++){

		vec gamma(3,fill::randu);
		gamma = 2*gamma;
		testCorrelationFunction.setGamma(gamma);

		rowvec x1(3,fill::randu);
		rowvec x2(3,fill::randu);
		rowvec diffDirection(3,fill::randu);

		rowvec x2p = x2 + eps*diffDirection;
		rowvec x2m = x2 - eps*diffDirection;

		double Rp = testCorrelationFunction.computeCorrelation(x1,x2p);
		double Rm = testCorrelationFunction.computeCorrelation(x1,x2m);
		double Rdot = testCorrelationFunction.computeCorrelationDot(x1, x2, diffDirection);

		double fdValue = (Rp - Rm)/(2.0*eps);
		double error = fabs(fdValue - Rdot);
		//		printTwoScalars(fdValue, Rdot);
		EXPECT_LT(error,10E-6);

	}

}

//TEST_F(ExpCorrelationFunctionTest, testComputeCorrelationDotDot){
//
//	testCorrelationFunction.initialize();
//	const double eps = 10E-010;
//
//	for(unsigned int i=0; i<1000; i++){
//
//		vec gamma(3,fill::randu);
//		gamma = 2*gamma;
//		testCorrelationFunction.setGamma(gamma);
//
//		rowvec x1(3,fill::randu);
//		rowvec x2(3,fill::randu);
//		rowvec diffDirection1(3,fill::randu);
//		rowvec diffDirection2(3,fill::randu);
//
//		rowvec x2p = x2 + eps*diffDirection2;
//		rowvec x2m = x2 - eps*diffDirection2;
//
//		double Rp = testCorrelationFunction.computeCorrelationDot(x1,x2p,diffDirection1);
//		double Rm = testCorrelationFunction.computeCorrelationDot(x1,x2m,diffDirection1 );
//		double Rdot = testCorrelationFunction.computeCorrelationDotDot(x1, x2, diffDirection1,diffDirection2);
//
//		double fdValue = (Rp - Rm)/(2.0*eps);
//		double error = fabs(fdValue - Rdot);
//		//		printTwoScalars(fdValue, Rdot);
//		EXPECT_LT(error,10E-4);
//
//	}
//}
//
//TEST_F(ExpCorrelationFunctionTest, testComputeDifferentitaedCorrelationDotAtZero){
//
//	testCorrelationFunction.initialize();
//
//
//	const double eps = 10E-010;
//
//
//
//	vec gamma(3,fill::randu);
//	gamma = 2*gamma;
//	testCorrelationFunction.setGamma(gamma);
//
//	rowvec x1(3,fill::randu);
//	rowvec x2 = x1 + 0.0000001;
//	rowvec x3 = x1;
//	rowvec diffDirection1(3,fill::randu);
//	rowvec diffDirection2(3,fill::randu);
//
//	double Rdot1 = testCorrelationFunction.computeCorrelationDot(x1, x2, diffDirection);
//
//	printScalar(Rdot1);
//
//}








#endif
