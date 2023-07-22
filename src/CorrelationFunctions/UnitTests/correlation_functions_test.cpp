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

#include "../INCLUDE/correlation_functions.hpp"
#include "../INCLUDE/exponential_correlation_function.hpp"
#include "../INCLUDE/gaussian_correlation_function.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include<gtest/gtest.h>



mat generateDataMatrixForTestingCorrelationFunctions(unsigned int N, unsigned int d){

	mat dataMatrix(N,d);
	for(unsigned int i=0; i<N; i++){

		for(unsigned int j=0; j<d; j++){

			dataMatrix(i,j) = sin(i*j) + 2.0*cos(i+j);

		}


	}
	return dataMatrix;

}


class CorrelationFunctionsTest : public ::testing::Test {
protected:
	void SetUp() override {


		testCorrelationFunction.setDimension(0);

	}

	void TearDown() override {}



	ExponentialCorrelationFunction testCorrelationFunction;
	GaussianCorrelationFunction testCorrelationFunctionGEK;



};

TEST_F(CorrelationFunctionsTest, testinitialize){


	ASSERT_FALSE(testCorrelationFunction.isInputSampleMatrixSet());

}

TEST_F(CorrelationFunctionsTest, testSetInputSampleMatrix){

	unsigned int N = 100;
	unsigned dim = 5;
	mat testInput(N,dim,fill::randu);

	testCorrelationFunction.setInputSampleMatrix(testInput);
	ASSERT_TRUE(testCorrelationFunction.isInputSampleMatrixSet());

}

TEST_F(CorrelationFunctionsTest, testcomputeCorrelationMatrix){

	unsigned int N = 5;
	unsigned dim = 5;
	mat testInput = generateDataMatrixForTestingCorrelationFunctions(N,dim);

	vec theta(dim); theta.fill(0.1);
	vec gamma(dim); gamma.fill(2.0);

	testCorrelationFunction.setDimension(dim);
	testCorrelationFunction.setTheta(theta);
	testCorrelationFunction.setGamma(gamma);
	testCorrelationFunction.setInputSampleMatrix(testInput);
	testCorrelationFunction.computeCorrelationMatrix();

	mat R = testCorrelationFunction.getCorrelationMatrix();
	double valueToExpect = 0.67288068346;
	double error = fabs(R(0,1) - valueToExpect);
	EXPECT_LT(error, 10E-08);

}

//TEST_F(CorrelationFunctionsTest, testcomputeCorrelationMatrixBiQuadSpline){
//
//	unsigned int N = 10;
//	unsigned dim = 5;
//	mat testInput = generateDataMatrixForTestingCorrelationFunctions(N,dim);
//
//	vec theta(dim); theta.fill(0.1);
//
//
//	testCorrelationFunctionBiQuadraticSpline.setHyperParameters(theta);
//	testCorrelationFunctionBiQuadraticSpline.setInputSampleMatrix(testInput);
//	testCorrelationFunctionBiQuadraticSpline.computeCorrelationMatrix();
//
//	mat R = testCorrelationFunctionBiQuadraticSpline.getCorrelationMatrix();
//
//
//	testCorrelationFunction2.corrbiquadspline_kriging(testInput, theta);
//
//	double valueToExpect = 0.60924674573;
//	double error = fabs(R(0,1) - valueToExpect);
//	EXPECT_LT(error, 10E-08);
//
//}



//TEST_F(CorrelationFunctionsTest, testGEK_compute_dCorrelationMatrixdxi){
//
//	unsigned int N = 3;
//	unsigned dim = 3;
//	mat testInput = generateDataMatrixForTestingCorrelationFunctions(N,dim);
//
//	vec theta(dim); theta.fill(0.1);
//	testCorrelationFunctionGEK.setHyperParameters(theta);
//	testCorrelationFunctionGEK.setInputSampleMatrix(testInput);
//	mat dPsidx0 = testCorrelationFunctionGEK.compute_dCorrelationMatrixdxi(0);
//
//
//
//
//	testCorrelationFunctionGEK.computeCorrelationMatrixDotForrester();
//	mat Rdot = testCorrelationFunctionGEK.getCorrelationMatrixDot();
//
//	testCorrelationFunction2.corrgaussian_gekriging(testInput, theta);
//
//
//}
//
//TEST_F(CorrelationFunctionsTest, testcomputeCorrelationMatrixDot){
//
//	unsigned int N = 3;
//	unsigned dim = 3;
//	mat testInput = generateDataMatrixForTestingCorrelationFunctions(N,dim);
//
//	vec theta(dim); theta.fill(0.1);
//	testCorrelationFunctionGEK.setHyperParameters(theta);
//	testCorrelationFunctionGEK.setInputSampleMatrix(testInput);
//	testCorrelationFunctionGEK.computeCorrelationMatrixDot();
//
//
//
//
//
//}

class GaussianCorrelationFunctionsTest : public ::testing::Test {
protected:
	void SetUp() override {


		testCorrelationFunction.setDimension(3);

	}

	//  void TearDown() override {}


	GaussianCorrelationFunction testCorrelationFunction;



};

TEST_F(GaussianCorrelationFunctionsTest, testSetInputMatrix){

	ASSERT_FALSE(testCorrelationFunction.isInputSampleMatrixSet());
	mat dataMatrix = generateDataMatrixForTestingCorrelationFunctions(10,3);
	testCorrelationFunction.setInputSampleMatrix(dataMatrix);
	ASSERT_TRUE(testCorrelationFunction.isInputSampleMatrixSet());
}

TEST_F(GaussianCorrelationFunctionsTest, testComputeCorrelation){

	testCorrelationFunction.initialize();
	rowvec x1(3,fill::randu);
	rowvec x2(3,fill::randu);

	double R = testCorrelationFunction.computeCorrelation(x1,x2);

	double Rexpected = exp(- ((x1(0)-x2(0))*(x1(0)-x2(0)) + (x1(1)-x2(1))*(x1(1)-x2(1)) + (x1(2)-x2(2))*(x1(2)-x2(2)) ));
	double error = fabs(Rexpected - R);
	EXPECT_LT(error,10E-10);

}

TEST_F(GaussianCorrelationFunctionsTest, testComputeCorrelationDot){

	testCorrelationFunction.initialize();
	rowvec x1(3,fill::randu);
	rowvec x2(3,fill::randu);
	rowvec diffDirection(3,fill::randu);

	double R0 = testCorrelationFunction.computeCorrelation(x1,x2);
	double Rdot = testCorrelationFunction.computeCorrelationDot(x1, x2, diffDirection);
	rowvec x2pert = x2 + 0.0001*diffDirection;
	double Rplus = testCorrelationFunction.computeCorrelation(x1,x2pert);
	double fdValue = (Rplus-R0)/0.0001;
	double error = fabs(fdValue - Rdot);
	EXPECT_LT(error,10E-4);

}

TEST_F(GaussianCorrelationFunctionsTest, testComputeCorrelationDotDot){

	testCorrelationFunction.initialize();
	rowvec x1(3,fill::randu);
	rowvec x2(3,fill::randu);
	rowvec firstdiffDirection(3,fill::randu);
	rowvec seconddiffDirection(3,fill::randu);

	double Rdot0 = testCorrelationFunction.computeCorrelationDot(x1,x2, firstdiffDirection);
	double Rdotdot = testCorrelationFunction.computeCorrelationDotDot(x1, x2, firstdiffDirection,seconddiffDirection);
	rowvec x2pert = x2 + 0.0001*seconddiffDirection;
	double Rdotplus = testCorrelationFunction.computeCorrelationDot(x1,x2pert,firstdiffDirection);
	double fdValue = (Rdotplus-Rdot0)/0.0001;
	double error = fabs(fdValue - Rdotdot);
	EXPECT_LT(error,10E-4);

}

