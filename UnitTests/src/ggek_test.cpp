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

#include "ggek.hpp"
#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "standard_test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef GGEK_MODEL_TEST

class GGEKModelTest: public ::testing::Test {
protected:
	void SetUp() override {
		vec ub(2);
		vec lb(2);
		ub.fill(6.0);
		lb.fill(-6.0);
		boxConstraints.setBounds(lb,ub);

		testModel.setBoxConstraints(boxConstraints);
		testModel.setDimension(2);
		testModel.setName("himmelblauModel");

		himmelblauFunction.function.filenameTestData = "himmelblau.csv";
		himmelblauFunction.function.numberOfTrainingSamples = 50;
		himmelblauFunction.function.filenameTestData = "himmelblauTest.csv";
		himmelblauFunction.function.numberOfTestSamples = 100;

		N = himmelblauFunction.function.numberOfTrainingSamples;
		dim = 2;

		alpine02Function.function.filenameTrainingData = "alpine02.csv";
		alpine02Function.function.numberOfTrainingSamples = 50;
		alpine02Function.function.filenameTestData = "alpine02Test.csv";
		alpine02Function.function.numberOfTestSamples = 50;

	}

	void TearDown() override {}

	GGEKModel testModel;

	HimmelblauFunction himmelblauFunction;
	Alpine02_5DFunction alpine02Function;

	Bounds boxConstraints;

	unsigned int N;
	unsigned int dim;

};




TEST_F(GGEKModelTest, constructor) {


	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModel.ifNormalized);
	ASSERT_FALSE(testModel.ifHasTestData);


}

TEST_F(GGEKModelTest, readData) {

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();

	mat dataRead = testModel.getRawData();

	ASSERT_TRUE(testModel.getDimension() == dim);
	ASSERT_TRUE(dataRead.n_cols == 2*dim+1);
	ASSERT_TRUE(dataRead.n_rows == N);
	ASSERT_TRUE(testModel.ifDataIsRead);
}


TEST_F(GGEKModelTest, normalizeData) {

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	mat X = testModel.getX();

	ASSERT_TRUE(X.n_rows == N);
	ASSERT_TRUE(X.n_cols == dim);
	ASSERT_TRUE(testModel.ifNormalized);

}

TEST_F(GGEKModelTest, initializeSurrogateModel) {

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	//	testModel.printSurrogateModel();

	ASSERT_TRUE(testModel.ifInitialized);


}

TEST_F(GGEKModelTest, generateSampleWeights) {


	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

//	testModel.setDisplayOn();
	testModel.generateSampleWeights();

	vec w = testModel.getSampleWeightsVector();

	ASSERT_TRUE(max(w) == 1.0);
	ASSERT_TRUE(w.size() == N);


}

TEST_F(GGEKModelTest, calculateIndicesOfSamplesWithActiveDerivatives) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.calculateIndicesOfSamplesWithActiveDerivatives();

}

TEST_F(GGEKModelTest, generateWeightingMatrix) {


	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();

	testModel.ifVaryingSampleWeights = true;

	testModel.generateWeightingMatrix();

	mat W = testModel.getWeightMatrix();

	unsigned int Ndot = testModel.getNumberOfSamplesWithActiveGradients();
	N    = testModel.getNumberOfSamples();

	ASSERT_TRUE(W.n_rows == N+Ndot);
	ASSERT_TRUE(W.n_cols == N+Ndot);

}

TEST_F(GGEKModelTest, generateRhsForRBFs) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();
	testModel.ifVaryingSampleWeights = true;
	testModel.generateWeightingMatrix();
	testModel.generateRhsForRBFs();



}


TEST_F(GGEKModelTest, calculatePhiMatrix) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	//	testModel.setDisplayOn();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();

	testModel.initializeSurrogateModel();

	testModel.generateWeightingMatrix();

	testModel.calculatePhiMatrix();

	mat Phi = testModel.getPhiMatrix();

	unsigned int Ndot = testModel.getNumberOfSamplesWithActiveGradients();
	N    = testModel.getNumberOfSamples();

	ASSERT_TRUE(Phi.n_cols == N + Ndot);
	ASSERT_TRUE(Phi.n_rows == N + Ndot);


	ASSERT_TRUE(testModel.checkPhiMatrix());

}


TEST_F(GGEKModelTest, assembleLinearSystem) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();


	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();
	testModel.initializeSurrogateModel();

	testModel.assembleLinearSystem();

	mat Phi = testModel.getPhiMatrix();


	ASSERT_TRUE(Phi.n_cols > 0);
	ASSERT_TRUE(Phi.n_rows > 0);

}




TEST_F(GGEKModelTest, train) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();


	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.train();

	ASSERT_TRUE(testModel.checkResidual());
	ASSERT_TRUE(testModel.ifModelTrainingIsDone);


}

TEST_F(GGEKModelTest, interpolate) {

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = true;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();


	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.train();

	rowvec x(2);
	x(0) = 0.1;
	x(1) = 0.01;
	double fVal = testModel.interpolate(x);

	ASSERT_TRUE(testModel.ifModelTrainingIsDone);
}





TEST_F(GGEKModelTest, tryOnTestData) {

	alpine02Function.function.numberOfTrainingSamples = 50;
	alpine02Function.function.numberOfTestSamples = 50;
//	alpine02Function.function.ifSomeAdjointsAreLeftBlank = true;
	alpine02Function.function.generateTrainingSamplesWithAdjoints();

	alpine02Function.function.generateTestSamples();

	boxConstraints = alpine02Function.function.boxConstraints;

	testModel.setBoxConstraints(boxConstraints);
	testModel.setName("Alpine02Model");

	testModel.setDimension(5);

	testModel.setNameOfInputFile(alpine02Function.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
//	testModel.setDisplayOn();
//	testModel.ifVaryingSampleWeights = true;

	testModel.initializeSurrogateModel();

	testModel.train();
	ASSERT_TRUE(testModel.checkPhiMatrix());
	ASSERT_TRUE(testModel.checkResidual());


	testModel.setNameOfInputFileTest(alpine02Function.function.filenameTestData);

	testModel.readDataTest();
	testModel.normalizeDataTest();

	testModel.tryOnTestData();
//	testModel.printGeneralizationError();

	double MSE = testModel.generalizationError;

	ASSERT_TRUE(MSE>0.0);

}




TEST_F(GGEKModelTest, determineThetaCoefficientForDualBasis) {


	alpine02Function.function.generateTrainingSamplesWithAdjoints();

	boxConstraints =  alpine02Function.function.boxConstraints;
	testModel.setName("Alpine02");
	testModel.setDimension(5);
	testModel.setBoxConstraints(boxConstraints);


	testModel.setNameOfInputFile(alpine02Function.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.trainTheta();
	testModel.determineThetaCoefficientForDualBasis();

	ASSERT_TRUE(testModel.ifThetaFactorOptimizationIsDone);


}


TEST_F(GGEKModelTest, updateModelWithNewData) {

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();


	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.train();

	unsigned int howManySamples = testModel.getNumberOfSamples();

	ASSERT_TRUE( howManySamples == N);

	double x[2]; x[0] = 1.34; x[1] = 0.88;

	double f = himmelblauFunction.function.func_ptr(x);

	rowvec newsample(3,fill::randu);
	newsample(0) = x[0];
	newsample(1) = x[1];
	newsample(2) = f;

	testModel.addNewSampleToData(newsample);

	howManySamples = testModel.getNumberOfSamples();

	ASSERT_TRUE(howManySamples == N+1);

	mat PhiNew = testModel.getPhiMatrix();

	ASSERT_TRUE(PhiNew.n_cols == 2*N+1);
	ASSERT_TRUE(PhiNew.n_rows == 2*N+1);

	Bounds boxConstraints = himmelblauFunction.function.boxConstraints;
	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	rowvec xp(2); xp(0) = x[0]; xp(1) = x[1];

	xp = xp + 0.0001;
	xp = normalizeRowVector(xp,lb,ub);
	double ftilde = testModel.interpolate(xp);

	double error  = fabs(ftilde - f)/f;
	EXPECT_LT(error, 0.05);

}

#endif
