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


#include "aggregation_model.hpp"
#include "auxiliary_functions.hpp"
#include "test_functions.hpp"
#include "random_functions.hpp"
#include "polynomials.hpp"
#include <vector>
#include <armadillo>

#include<gtest/gtest.h>

#define TESTAGGREGATION
#ifdef TESTAGGREGATION


TEST(testAggrregationModel, testfindNearestNeighbor){

	unsigned int dim = 2;
	unsigned int N = 100;
	mat samples(N,2*dim+1,fill::randu);
	saveMatToCVSFile(samples,"AggregationTest.csv");


	AggregationModel testModel("AggregationTest");

	testModel.readData();
	Bounds boxConstraints(dim);
	boxConstraints.setBounds(0.0, 1.0);


	testModel.setBoxConstraints(boxConstraints);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();


	rowvec x = testModel.getRowX(1);
	unsigned int NNIndex = testModel.findNearestNeighbor(x);
	ASSERT_EQ(NNIndex,1);
	x = testModel.getRowX(45);
	x(0) += 0.0001;
	x(1) -= 0.0001;
	NNIndex = testModel.findNearestNeighbor(x);
	ASSERT_EQ(NNIndex,45);
	remove("AggregationTest.csv");

}


TEST(testAggrregationModel, testinterpolateWithLinearFunction){

	unsigned int N = 100;
	unsigned int dim = 2;

	mat samples(N,2*dim+1);

	/* we construct first test data using a linear function 2*x1 + x2, including gradients */
	for (unsigned int i=0; i<N; i++){
		rowvec x(2*dim+1);
		x(0) = generateRandomDouble(0.0,1.0);
		x(1) = generateRandomDouble(0.0,1.0);

		x(2) = 2*x(0) + x(1);
		x(3) = 2.0;
		x(4) = 1.0;
		samples.row(i) = x;

	}


	saveMatToCVSFile(samples,"AggregationTest.csv");
	AggregationModel testModel("AggregationTest");

	testModel.readData();

	testModel.setBoxConstraints(0.0,1.0);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.setRho(0.0);

	rowvec xTest(2);
	xTest.fill(0.25);
	rowvec xTestNotNormalized = 2.0*xTest;

	double testValue = testModel. interpolate(xTest);
	double exactValue = 2*xTestNotNormalized(0) + xTestNotNormalized(1);
	double error = fabs(testValue - exactValue);
	EXPECT_LT(error, 10E-010);

	remove("AggregationTest.csv");

}


TEST(testAggrregationModel, determineOptimalL1NormWeights){

	unsigned int N = 200;
	mat samples(N,7);

	/* y = f(x1,x2,x3) = x1^2 */
	for (unsigned int i=0; i<N; i++){
		rowvec x(7);
		x(0) = generateRandomDouble(0.0,1.0);
		x(1) = generateRandomDouble(0.0,1.0);
		x(2) = generateRandomDouble(0.0,1.0);

		x(3) = 2*x(0)*x(0);
		x(4) = 2.0*x(0);
		x(5) = 0.0;
		x(6) = 0.0;
		samples.row(i) = x;

	}


	saveMatToCVSFile(samples,"AggregationTest.csv");


	AggregationModel testModel("AggregationTest");



	testModel.readData();

	testModel.setBoxConstraints(0.0, 1.0);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.setNumberOfTrainingIterations(1000);

	testModel.determineOptimalL1NormWeights();

	vec optimalWeights = testModel.getL1NormWeights();
	EXPECT_LT(optimalWeights(1), 0.2);
	EXPECT_LT(optimalWeights(2), 0.2);
	remove("AggregationTest.csv");

}


#endif















