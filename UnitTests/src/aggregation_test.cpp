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


#include "aggregation_model.hpp"
#include "auxiliary_functions.hpp"
#include "standard_test_functions.hpp"
#include "random_functions.hpp"
#include "polynomials.hpp"
#include "test_defines.hpp"
#include <vector>
#include <armadillo>

#include<gtest/gtest.h>

#ifdef AGGREGATION_TEST

class AggregationModelTest: public ::testing::Test {
protected:
	void SetUp() override {}

	void TearDown() override {}


	AggregationModel testModel;
	HimmelblauFunction himmelblauFunction;

};


TEST_F(AggregationModelTest, tryOnTestData){

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	himmelblauFunction.function.generateTestSamplesCloseToTrainingSamples();

	testModel.setDimension(2);
	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	Bounds boxConstraints;
	boxConstraints = himmelblauFunction.function.boxConstraints;
	testModel.setBoxConstraints(boxConstraints);


	testModel.setDisplayOn();
	testModel.readData();
	testModel.normalizeData();


	testModel.initializeSurrogateModel();
	testModel.setNumberOfTrainingIterations(20000);
	testModel.train();


	testModel.setNameOfInputFileTest(himmelblauFunction.function.filenameTestData);
	testModel.readDataTest();
	testModel.normalizeDataTest();

	testModel.tryOnTestData();
	testModel.printGeneralizationError();

	abort();


}




TEST_F(AggregationModelTest, constructor){

	ASSERT_FALSE(testModel.ifDataIsRead);

}


TEST_F(AggregationModelTest, readData){

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	testModel.setDimension(2);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	//	testModel.setDisplayOn();
	testModel.readData();

	ASSERT_TRUE(testModel.ifDataIsRead);

}

TEST_F(AggregationModelTest, initializeSurrogateModel){

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setDimension(2);
	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	Bounds boxConstraints;
	boxConstraints = himmelblauFunction.function.boxConstraints;
	testModel.setBoxConstraints(boxConstraints);


	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.normalizeData();


	testModel.initializeSurrogateModel();

	ASSERT_TRUE(testModel.ifInitialized);
}


TEST_F(AggregationModelTest, findNearestNeighbor){

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setDimension(2);
	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	Bounds boxConstraints;
	boxConstraints = himmelblauFunction.function.boxConstraints;
	testModel.setBoxConstraints(boxConstraints);


	//	testModel.setDisplayOn();
	testModel.readData();
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



}

//TEST_F(AggregationModelTest, determineOptimalL1NormWeights){
//
//	unsigned int N = 200;
//	unsigned int d = 3;
//	mat samples(N,2*d+1);
//
//	/* y = f(x1,x2,x3) = x1^2 */
//	for (unsigned int i=0; i<N; i++){
//		rowvec x(2*d+1);
//		x(0) = generateRandomDouble(0.0,1.0);
//		x(1) = generateRandomDouble(0.0,1.0);
//		x(2) = generateRandomDouble(0.0,1.0);
//
//		x(3) = 2*x(0)*x(0);
//
//		x(4) = 2.0*x(0);
//		x(5) = 0.0;
//		x(6) = 0.0;
//		samples.row(i) = x;
//
//	}
//
//	samples.save("AggregationTest.csv", csv_ascii);
//
//
//	testModel.setDimension(3);
//	vec lb(3); vec ub(3);
//	lb.fill(0.0);
//	ub.fill(1.0);
//
//	Bounds boxConstraints;
//	boxConstraints.setBounds(lb,ub);
//
//	testModel.setBoxConstraints(boxConstraints);
//
//	testModel.setNameOfInputFile("AggregationTest.csv");
//	//	testModel.setDisplayOn();
//
//	testModel.readData();
//	testModel.normalizeData();
//	testModel.initializeSurrogateModel();
//	testModel.setNumberOfTrainingIterations(1000);
//
//
//	testModel.determineOptimalL1NormWeights();
//
//	vec optimalWeights = testModel.getL1NormWeights();
//	EXPECT_LT(optimalWeights(1), 0.2);
//	EXPECT_LT(optimalWeights(2), 0.2);
//	remove("AggregationTest.csv");
//
//}


TEST_F(AggregationModelTest, train){

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setDimension(2);
	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	Bounds boxConstraints;
	boxConstraints = himmelblauFunction.function.boxConstraints;
	testModel.setBoxConstraints(boxConstraints);


	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.normalizeData();


	testModel.initializeSurrogateModel();
	testModel.train();

	ASSERT_TRUE(testModel.ifModelTrainingIsDone);

}





#endif















