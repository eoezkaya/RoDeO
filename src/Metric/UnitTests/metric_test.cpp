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

#include "../INCLUDE/metric.hpp"
#include "../../TestFunctions/INCLUDE/test_functions.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include<gtest/gtest.h>


double lowDimensionalTestFunction(rowvec x){

	return x(0)*x(0) + 0.1 * x(1)*x(1);

}

mat generateDataUsinglowDimensionalTestFunction(unsigned int N){

	mat X(N,5,fill::randu);
	mat output(N,6);
	for(unsigned int i=0; i<N; i++){

		rowvec xp = X.row(i);
		double f = lowDimensionalTestFunction(xp);
		addOneElement(xp,f);
		output.row(i) = xp;
	}


	return output;
}



class MetricTest : public ::testing::Test {
protected:
	void SetUp() override {

		unsigned int dim = 5;
		testNorm.initialize(dim);


	}

	//  void TearDown() override {}

	WeightedL1Norm testNorm;


};
//
//TEST_F(MetricTest, testinitialize){
//
//	ASSERT_FALSE(testNorm.isNumberOfTrainingIterationsSet());
//	ASSERT_FALSE(testNorm.isTrainingDataSet());
//	ASSERT_FALSE(testNorm.isValidationDataSet());
//
//	vec w = testNorm.getWeights();
//
//	double error = fabs(w(0) - 1.0/5) + fabs(w(1) - 1.0/5) + fabs(w(2) - 1.0/5) + fabs(w(3) - 1.0/5) + fabs(w(4) - 1.0/5);
//
//	EXPECT_LT(error,10E-10);
//	ASSERT_TRUE(testNorm.getDimension() == 5);
//
//}
//
//
//TEST_F(MetricTest, testsetTrainingData){
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//	testNorm.setTrainingData(testData);
//
//	assert(testNorm.isTrainingDataSet());
//
//}
//
//TEST_F(MetricTest, testsetValidationData){
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//	testNorm.setValidationData(testData);
//
//	assert(testNorm.isValidationDataSet());
//
//}
//
//TEST_F(MetricTest, testcalculateNorm){
//
//	rowvec x(5);
//	x(0) = 1.0; x(1) = -2.0; x(2) = 3.0; x(3) = -4.0; x(4) = 5.0;
//	double norm = testNorm.calculateNorm(x);
//
//	double check = 0.2*15;
//	double error = fabs(norm-check);
//	EXPECT_LT(error,10E-10);
//
//
//}
//
//TEST_F(MetricTest, testfindNearestNeighbor){
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//
//	rowvec randomRow(dim+1,fill::randu);
//
//	testData.row(17) = randomRow;
//
//	testNorm.setTrainingData(testData);
//	rowvec xp(dim);
//	xp = randomRow.head(dim);
//	xp +=0.000001;
//
//	int index = testNorm.findNearestNeighbor(xp);
//
//	ASSERT_TRUE(index == 17);
//}
//
//
//TEST_F(MetricTest, testinterpolateByNearestNeighbour){
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//
//	rowvec randomRow(dim+1,fill::randu);
//
//	testData.row(17) = randomRow;
//
//	testNorm.setTrainingData(testData);
//	rowvec xp(dim);
//	xp = randomRow.head(dim);
//	xp +=0.000001;
//
//	double value = testNorm.interpolateByNearestNeighbor(xp);
//	double error = fabs(value - randomRow(dim));
//	EXPECT_LT(error,10E-10);
//}
//
//
//TEST_F(MetricTest, testcalculateMeanL1ErrorOnData){
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//
//	rowvec randomRow1(dim+1,fill::randu);
//	rowvec randomRow2(dim+1,fill::randu);
//
//	testData.row(17) = randomRow1;
//	testData.row(22) = randomRow2;
//
//	testNorm.setTrainingData(testData);
//	rowvec xp(dim);
//
//	mat validationData(2,dim+1);
//	validationData.row(0) = randomRow1 + 0.0001;
//	validationData.row(1) = randomRow2 + 0.0001;
//
//	testNorm.setValidationData(validationData);
//
//	double L1error = testNorm.calculateMeanL1ErrorOnData();
//	double L1errorExpected = 0.0001;
//	double error = fabs(L1error - L1errorExpected);
//
//	EXPECT_LT(error, 10E-08);
//
//
//
//}
//
//
//
//TEST_F(MetricTest, testinitializeWeightedL1NormObject){
//
//	WeightedL1NormOptimizer testOptimizer;
//
//	unsigned int N = 100;
//	unsigned int dim = 5;
//	mat testData(N,dim+1, fill::randu);
//	mat validationData(N,dim+1, fill::randu);
//	testNorm.setTrainingData(testData);
//	testNorm.setValidationData(validationData);
//
//	testOptimizer.initializeWeightedL1NormObject(testNorm);
//
//}
//
//
//
//TEST_F(MetricTest, testfindOptimalWeights){
//
//
//	unsigned int numberOfTrainingSamples   = 100;
//	unsigned int numberOfValidationSamples = 100;
//	unsigned int dim = 5;
//
//
//	mat trainingData = generateDataUsinglowDimensionalTestFunction(numberOfTrainingSamples);
//	mat validationData = generateDataUsinglowDimensionalTestFunction(numberOfValidationSamples );
//
//
//	testNorm.setTrainingData(trainingData);
//	testNorm.setValidationData(validationData);
//
//	testNorm.findOptimalWeights();
//
//	vec weights = testNorm.getWeights();
//	EXPECT_TRUE(weights(0) > 0.5);
//
//}
//TEST_F(MetricTest, testfindOptimalWeightsParallel){
//
//
//	unsigned int numberOfTrainingSamples   = 100;
//	unsigned int numberOfValidationSamples = 100;
//	unsigned int dim = 5;
//	mat trainingData = generateDataUsinglowDimensionalTestFunction(numberOfTrainingSamples);
//	mat validationData = generateDataUsinglowDimensionalTestFunction(numberOfValidationSamples );
//
//
//	testNorm.setTrainingData(trainingData);
//	testNorm.setValidationData(validationData);
//	testNorm.setNumberOfThreads(2);
//
//	testNorm.findOptimalWeights();
//	vec weights = testNorm.getWeights();
//	EXPECT_TRUE(weights(0) > 0.5);
//
//}


