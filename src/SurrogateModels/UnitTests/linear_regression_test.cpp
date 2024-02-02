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
#include "../INCLUDE/linear_regression.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../../TestFunctions/INCLUDE/standard_test_functions.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include<gtest/gtest.h>



class LinearModelTest : public ::testing::Test {
protected:
	void SetUp() override {

		testFunction.function.numberOfTrainingSamples = 20;
		testFunction.function.numberOfTestSamples = 50;
		testFunction.function.filenameTrainingData = "linearTF1.csv";
		testFunction.function.filenameTestData = "linearTF1Test.csv";

		boxConstraints.setDimension(2);
		boxConstraints.setBounds(-6,6);

		testModel2D.setDimension(2);
		testModel2D.setNameOfInputFile(testFunction.function.filenameTrainingData);
		testModel2D.setNameOfInputFileTest(testFunction.function.filenameTestData);

		testModel2D.setBoxConstraints(boxConstraints);

	}

	void TearDown() override {}

	Bounds boxConstraints;

	LinearModel testModel2D;
	LinearTestFunction1 testFunction;


};


TEST_F(LinearModelTest, constructor) {

	ASSERT_FALSE(testModel2D.ifDataIsRead);
	ASSERT_FALSE(testModel2D.ifInitialized);
	ASSERT_FALSE(testModel2D.ifNormalized);
	ASSERT_TRUE(testModel2D.ifHasTestData);
	ASSERT_TRUE(testModel2D.getDimension() == 2);
	std::string filenameDataInput = testModel2D.getNameOfInputFile();
	ASSERT_TRUE(filenameDataInput == "linearTF1.csv");
}

TEST_F(LinearModelTest, readData) {

	testFunction.function.generateTrainingSamples();
	testModel2D.readData();
	ASSERT_TRUE(testModel2D.ifDataIsRead);
}

TEST_F(LinearModelTest, normalizeData) {

	testFunction.function.generateTrainingSamples();
	testModel2D.readData();
	testModel2D.normalizeData();
	ASSERT_TRUE(testModel2D.ifNormalized);
}

TEST_F(LinearModelTest, initializeSurrogateModel) {

	testFunction.function.generateTrainingSamples();
//	testModel2D.setDisplayOn();
	testModel2D.readData();
	testModel2D.normalizeData();
	testModel2D.initializeSurrogateModel();
	ASSERT_TRUE(testModel2D.ifInitialized);
}

TEST_F(LinearModelTest, testAllX) {

	testFunction.function.generateTrainingSamples();

	testModel2D.setRegularizationParameter(0.0);
//	testModel2D.setDisplayOn();
	testModel2D.readData();
	testModel2D.normalizeData();
	testModel2D.initializeSurrogateModel();


	mat testData(50,2,fill::randu);
	testData = testData*0.5;

	vec result = testModel2D.interpolateAll(testData);

	vec ub(2); ub.fill( 6);
	vec lb(2); lb.fill(-6);

	for(unsigned int i=0; i<50; i++){

		rowvec x = testData.row(i);
		x = normalizeVectorBack(x,lb,ub);
		double f = testFunction.function.func_ptr(x.memptr());
		double err = fabs(f - result(i));

		EXPECT_LT(err,10E-2);

	}



}



