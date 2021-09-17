/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "metric.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>

TEST(testMetric, testinitalize){

	WeightedL1Norm normTest;

	normTest.initialize(5);

	unsigned int dim = normTest.getDimension();

	ASSERT_EQ(dim, 5);

}


TEST(testMetric, testfindNearestNeighborL1){

	mat X(100,5, fill::randu);

	rowvec xp(5,fill::randu);
	xp+=0.000001;

	X.row(11) = xp;
	unsigned int indx = findNearestNeighborL1(xp, X);

	ASSERT_EQ(indx, 11);

}

TEST(testMetric, testcalculateL1norm){

	rowvec xp(4); xp(0) = -1; xp(1) = 1; xp(2) = -2; xp(3) = 2;

	double normL1 = calculateL1norm(xp);
	double error = fabs(normL1-6.0);
	EXPECT_LT(error, 10E-10);

}


TEST(testMetric, testinterpolateByNearestNeighbour){

	unsigned int N = 100;
	unsigned int dim = 4;
	unsigned int indexToTest = 3;

	mat testData(N,dim+1, fill::randu);
	mat X = testData.submat(0,0,N-1,dim-1);

	vec w(dim, fill::randu);
	WeightedL1Norm normTest(w);
	rowvec xp = X.row(indexToTest);
	xp+= 0.0001;

	normTest.setTrainingData(testData);

	double y = normTest.interpolateByNearestNeighbour(xp);
	double error = fabs(y-testData(indexToTest,dim));
	EXPECT_LT(error, 10E-10);


}

TEST(testMetric, testcalculateMeanSquaredErrorOnData){

	unsigned int Ntraining   = 1000;
	unsigned int Nvalidation = 20;
	mat dataTraining(Ntraining,3);
	mat dataValidation(Nvalidation,3);


	for(unsigned int i=0; i<Ntraining; i++){

		rowvec x(3);
		double xp[2];
		x(0) = generateRandomDouble(-6.0,6.0);
		x(1) = generateRandomDouble(-6.0,6.0);
		xp[0] = x(0);
		xp[1] = x(1);
		x(2) = Himmelblau(xp);
		dataTraining.row(i) = x;
	}

	for(unsigned int i=0; i<Nvalidation; i++){

		rowvec x(3);
		double xp[2];
		x(0) = generateRandomDouble(-6.0,6.0);
		x(1) = generateRandomDouble(-6.0,6.0);
		xp[0] = x(0);
		xp[1] = x(1);
		x(2) = Himmelblau(xp);
		dataValidation.row(i) = x;
	}
	vec w(2, fill::ones);


	WeightedL1Norm normTest(w);

	normTest.setTrainingData(dataTraining);
	normTest.setValidationData(dataValidation);

	double MSE = normTest.calculateMeanSquaredErrorOnData();

	EXPECT_LT(MSE, 10000);

}


TEST(testMetric, testfindOptimalWeights){

	unsigned int NTraining   = 100;
	unsigned int NValidation = 20;
	unsigned int dim = 5;
	mat trainingData(NTraining,dim+1, fill::randu);
	mat validationData(NValidation,dim+1, fill::randu);

	/* we have a simple function y=f(x_1,x_2,x_3,x_4, ...) = x_1^2 + x_2^2 */
	for(unsigned int i=0; i<NTraining; i++){

		trainingData(i,dim) = trainingData(i,0)* trainingData(i,0) +  trainingData(i,1)* trainingData(i,1);

	}
	for(unsigned int i=0; i<NValidation; i++ ){

		validationData(i,dim) = validationData(i,0)* validationData(i,0) + validationData(i,1)* validationData(i,1);

	}


	WeightedL1Norm normTest(dim);
	normTest.setTrainingData(trainingData);
	normTest.setValidationData(validationData);

	normTest.ifDisplay = false;
	normTest.setNumberOfTrainingIterations(5000);
	normTest.findOptimalWeights();

	vec w = normTest.getWeights();

	EXPECT_GT(w(0), 1.0/dim);
	EXPECT_GT(w(1), 1.0/dim);


}


