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

#include "tgek.hpp"
#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "standard_test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef TEST_TANGENT_MODEL

class TGEKModelTest: public ::testing::Test {
protected:
	void SetUp() override {
		ub = zeros<vec>(2);
		ub.fill(6.0);
		lb = zeros<vec>(2);
		lb.fill(-6.0);

		Bounds boxConstraints(lb,ub);
		testModel.setBoxConstraints(boxConstraints);
		testModel.setDimension(2);


	}

	void TearDown() override {


	}

	void generate1DTestFunctionDataForTGEKModel(unsigned int N, unsigned int NTest) {


		oneDimensionalTestFunction.function.filenameTrainingData = "trainingSamples1DFunctionTGEK.csv";
		oneDimensionalTestFunction.function.numberOfTrainingSamples = N;
		oneDimensionalTestFunction.function.generateTrainingSamplesWithTangents();

		oneDimensionalTestFunction.function.filenameTestData = "testSamples1DFunctionTGEK.csv";
		oneDimensionalTestFunction.function.numberOfTestSamples  = NTest;
		oneDimensionalTestFunction.function.generateTestSamples();

		trainingData = oneDimensionalTestFunction.function.trainingSamples;
		testData = oneDimensionalTestFunction.function.testSamples;

	}

	void generate2DHimmelblauDataForTGEKModel(unsigned int N) {


		himmelblauFunction.function.filenameTrainingData ="trainingSamplesHimmelblauTGEK.csv";
		himmelblauFunction.function.numberOfTrainingSamples = N;
		himmelblauFunction.function.generateTrainingSamplesWithTangents();

		himmelblauFunction.function.filenameTestData = "testSamplesHimmelblauTGEK.csv";
		himmelblauFunction.function.numberOfTestSamples  = N;
		himmelblauFunction.function.generateTestSamples();

		trainingData = himmelblauFunction.function.trainingSamples;
		testData = himmelblauFunction.function.testSamples;

	}

	void generate2DHimmelblauDataForKrigingModel(unsigned int N) {

		himmelblauFunction.function.filenameTrainingData ="trainingSamplesHimmelblauTGEK.csv";
		himmelblauFunction.function.numberOfTrainingSamples = N;
		himmelblauFunction.function.generateTrainingSamples();

		himmelblauFunction.function.filenameTestData = "testSamplesHimmelblauTGEK.csv";
		himmelblauFunction.function.numberOfTestSamples  = N;
		himmelblauFunction.function.generateTestSamples();

		trainingData = himmelblauFunction.function.trainingSamples;
		testData = himmelblauFunction.function.testSamples;

	}

	mat trainingData;
	mat testData;
	TGEKModel testModel;
	HimmelblauFunction himmelblauFunction;
	NonLinear1DTestFunction1 oneDimensionalTestFunction;
	vec ub;
	vec lb;

};

TEST_F(TGEKModelTest, testConstructor) {

	TGEKModel testModel;
	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModel.ifNormalized);
	ASSERT_FALSE(testModel.ifHasTestData);
	ASSERT_FALSE(testModel.areGradientsOn());

}

TEST_F(TGEKModelTest, interpolateWithVarianceAfterDataAdd) {

	generate2DHimmelblauDataForTGEKModel(40);
	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
//	testModel.setDisplayOn();
	testModel.readData();


	testModel.normalizeData();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.initializeSurrogateModel();

	testModel.train();

	double ftilde;
	double ssqr;

	rowvec newsample(6);

	newsample(0) = -1.98257364824627;
	newsample(1) = -4.62142311872329;
	newsample(2) = 289.815465979252;
	newsample(3) = -131.120361660559;
	newsample(4) = 0.600378279235695;
	newsample(5) = 0.799716150782255;
	rowvec xp(2); xp(0) = newsample(0); xp(1) = newsample(1);

	rowvec xIn = normalizeRowVector(xp,lb,ub);

	testModel.interpolateWithVariance(xIn, &ftilde, &ssqr);

	testModel.addNewSampleToData(newsample);


	testModel.interpolateWithVariance(xIn, &ftilde, &ssqr);


	EXPECT_LT(ssqr, 10E-4);

}


TEST_F(TGEKModelTest, addNewSampleToData) {

	generate2DHimmelblauDataForTGEKModel(10);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();

	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	rowvec newSamplePoint(6, fill::randu);
	newSamplePoint(0) = -6;
	newSamplePoint(1) = 1.7;
	newSamplePoint(2) = 1.2;


	testModel.addNewSampleToData(newSamplePoint);

	mat data = testModel.getRawData();

	ASSERT_TRUE(data.n_rows == 11);

}

TEST_F(TGEKModelTest, calculatePhiMatrix) {

	generate2DHimmelblauDataForTGEKModel(3);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();


	testModel.normalizeData();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.initializeSurrogateModel();

	testModel.calculateIndicesOfSamplesWithActiveDerivatives();
	testModel.findIndicesOfDifferentiatedBasisFunctionLocations();

	testModel.generateWeightingMatrix();
	testModel.ifVaryingSampleWeights = false;

	testModel.calculatePhiMatrix();

	bool ifResultsAreOk = testModel.checkPhiMatrix();

	ASSERT_TRUE(ifResultsAreOk);

}

TEST_F(TGEKModelTest, interpolateWithVariance) {

	generate2DHimmelblauDataForTGEKModel(50);
	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();


	testModel.normalizeData();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.initializeSurrogateModel();

	testModel.train();

	rowvec x = trainingData.row(0);
	rowvec xp(2); xp(0) = x(0); xp(1) = x(1);
	double ftilde;
	double ssqr;

	rowvec xIn = normalizeRowVector(xp,lb,ub);
	testModel.interpolateWithVariance(xIn, &ftilde, &ssqr);

	EXPECT_LT(ssqr, 10E-4);

}



TEST_F(TGEKModelTest, generateSampleWeights) {

	unsigned int N = 10;
	generate2DHimmelblauDataForTGEKModel(N);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();
	testModel.generateSampleWeights();

	vec w = testModel.getSampleWeightsVector();

	ASSERT_TRUE(w.size() == N);
	double err = fabs(max(w) - 1.0);
	ASSERT_TRUE(err<10E-10);

}


TEST_F(TGEKModelTest, generateWeightingMatrix) {

	unsigned int N = 10;
	generate2DHimmelblauDataForTGEKModel(N);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.calculateIndicesOfSamplesWithActiveDerivatives();
	testModel.generateWeightingMatrix();

	mat W = testModel.getWeightMatrix();

	unsigned int Ndot = testModel.getNumberOfSamplesWithActiveGradients();

	ASSERT_TRUE(W.is_diagmat());

	for(unsigned int i=0; i<N+Ndot;i++){
		ASSERT_TRUE(W(i,i) >= 0.05);
	}
}

TEST_F(TGEKModelTest, calculateOutSampleError) {

	generate2DHimmelblauDataForTGEKModel(50);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();

	testModel.setNumberOfTrainingIterations(1000);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(5);
	testModel.train();

	testModel.setNameOfInputFileTest("testSamplesHimmelblauTGEK.csv");
	testModel.readDataTest();
	testModel.normalizeDataTest();

	double error = testModel.calculateOutSampleError();
	ASSERT_TRUE(error < 100.0);
}

TEST_F(TGEKModelTest, readData) {

	generate2DHimmelblauDataForTGEKModel(100);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");

	//	testModel.setDisplayOn();
	testModel.readData();

	mat dataRead = testModel.getRawData();
	bool ifIsEqual = checkMatrix(dataRead, trainingData, 10E-10);
	ASSERT_TRUE(ifIsEqual);

}

TEST_F(TGEKModelTest, normalizeData) {

	generate2DHimmelblauDataForTGEKModel(100);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();

	testModel.normalizeData();

	ASSERT_TRUE(testModel.ifNormalized);

}

TEST_F(TGEKModelTest, interpolate) {

	generate2DHimmelblauDataForTGEKModel(3);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();

	testModel.normalizeData();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.initializeSurrogateModel();

	testModel.updateAuxilliaryFields();
	rowvec x(2);
	x(0) = 0.3;
	x(1) = 0.25;


	double ftilde = testModel.interpolate(x);


}






TEST_F(TGEKModelTest, prepareTrainingDataForTheKrigingModel) {

	generate2DHimmelblauDataForTGEKModel(50);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.prepareTrainingDataForTheKrigingModel();

	mat trainingDataForKriging;
	trainingDataForKriging.load("auxiliaryData.csv", csv_ascii);

	mat traingDataCheck = trainingData.submat(0, 0, 49, 2);

	bool ifValuesAreOk = checkMatrix(traingDataCheck, trainingDataForKriging);
	ASSERT_TRUE(ifValuesAreOk);

}

TEST_F(TGEKModelTest, trainTheta) {

	generate2DHimmelblauDataForTGEKModel(3);
	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.normalizeData();
	testModel.setNumberOfTrainingIterations(1000);
	testModel.initializeSurrogateModel();
	testModel.trainTheta();

	vec theta = testModel.getHyperParameters();

	ASSERT_TRUE(theta.size() == 2);

}






TEST_F(TGEKModelTest, updateAuxilliaryFields) {

	generate2DHimmelblauDataForTGEKModel(50);
	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	//	testModel.setDisplayOn();
	testModel.updateAuxilliaryFields();

}

TEST_F(TGEKModelTest, train) {

	generate2DHimmelblauDataForTGEKModel(10);
	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.setNumberOfTrainingIterations(1000);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.train();

	ASSERT_TRUE(testModel.ifModelTrainingIsDone);

}


#endif