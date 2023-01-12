/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "tgek.hpp"
#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include "auxiliary_functions.hpp"
#include<gtest/gtest.h>



class TGEKModelTest: public ::testing::Test {
protected:
	void SetUp() override {

	}

	void TearDown() override {

	}

	void generate1DTestFunctionDataForTGEKModel(unsigned int N, unsigned int NTest) {

		TestFunction testFunctionHimmelblau("1DTestFunction", 1);

		testFunctionHimmelblau.setFunctionPointer(testFunction1D);
		testFunctionHimmelblau.setFunctionPointer(testFunction1DAdj);

		testFunctionHimmelblau.setBoxConstraints(0.0, 6.0);

		testFunctionHimmelblau.setNameFilenameTrainingData(
				"trainingSamplesHimmelblauTGEK.csv");
		testFunctionHimmelblau.setNumberOfTrainingSamples(N);
		testFunctionHimmelblau.generateSamplesInputTrainingData();
		testFunctionHimmelblau.generateTrainingSamplesWithTangents();

		testFunctionHimmelblau.setNameFilenameTestData(
				"testSamplesHimmelblauTGEK.csv");
		testFunctionHimmelblau.setNumberOfTestSamples(NTest);
		testFunctionHimmelblau.generateTestSamples();

		trainingData.load("trainingSamplesHimmelblauTGEK.csv", csv_ascii);
		testData.load("testSamplesHimmelblauTGEK.csv");

	}

	void generate2DHimmelblauDataForTGEKModel(unsigned int N) {

		TestFunction testFunctionHimmelblau("Himmelblau", 2);

		testFunctionHimmelblau.setFunctionPointer(Himmelblau);
		testFunctionHimmelblau.setFunctionPointer(HimmelblauAdj);

		testFunctionHimmelblau.setBoxConstraints(-6.0, 6.0);

		testFunctionHimmelblau.setNameFilenameTrainingData(
				"trainingSamplesHimmelblauTGEK.csv");
		testFunctionHimmelblau.setNumberOfTrainingSamples(N);
		testFunctionHimmelblau.generateSamplesInputTrainingData();
		testFunctionHimmelblau.generateTrainingSamplesWithTangents();

		testFunctionHimmelblau.setNameFilenameTestData(
				"testSamplesHimmelblauTGEK.csv");
		testFunctionHimmelblau.setNumberOfTestSamples(N);
		testFunctionHimmelblau.generateTestSamples();

		trainingData.load("trainingSamplesHimmelblauTGEK.csv", csv_ascii);
		testData.load("testSamplesHimmelblauTGEK.csv");

	}

	void generate2DHimmelblauDataForKrigingModel(unsigned int N) {

		TestFunction testFunctionHimmelblau("Himmelblau", 2);

		testFunctionHimmelblau.setFunctionPointer(Himmelblau);

		testFunctionHimmelblau.setBoxConstraints(-6.0, 6.0);

		testFunctionHimmelblau.setNameFilenameTrainingData(
				"trainingSamplesHimmelblauKriging.csv");
		testFunctionHimmelblau.setNumberOfTrainingSamples(N);
		testFunctionHimmelblau.generateSamplesInputTrainingData();
		testFunctionHimmelblau.generateTrainingSamples();

		testFunctionHimmelblau.setNameFilenameTestData(
				"testSamplesHimmelblauKriging.csv");
		testFunctionHimmelblau.setNumberOfTestSamples(N);
		testFunctionHimmelblau.generateTestSamples();

		trainingData.load("trainingSamplesHimmelblauKriging.csv", csv_ascii);
		testData.load("testSamplesHimmelblauKriging.csv");

	}

	mat trainingData;
	mat testData;
	TGEKModel testModel;

};

TEST_F(TGEKModelTest, testConstructor) {

	TGEKModel testModel;
	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModel.ifNormalized);
	ASSERT_FALSE(testModel.ifHasTestData);
	ASSERT_FALSE(testModel.areGradientsOn());

}

TEST_F(TGEKModelTest, generateSampleWeights) {

	generate2DHimmelblauDataForTGEKModel(10);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.generateSampleWeights();

	vec w = testModel.getSampleWeightsVector();

	ASSERT_TRUE(w.size() == 10);
	double err = fabs(max(w) - 1.0);
	ASSERT_TRUE(err<10E-10);

}


TEST_F(TGEKModelTest, generateWeightingMatrix) {

	generate2DHimmelblauDataForTGEKModel(5);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	testModel.generateWeightingMatrix();

	mat W = testModel.getWeightMatrix();

	for(unsigned int i=0; i<5;i++){

		ASSERT_TRUE(W(i,i) > 0.0);

	}


}

TEST_F(TGEKModelTest, calculateOutSampleError) {

	generate2DHimmelblauDataForTGEKModel(50);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);
	testModel.setNumberOfTrainingIterations(1000);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
//	testModel.setDisplayOn();

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
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);

	testModel.normalizeData();

	ASSERT_TRUE(testModel.ifNormalized);

}

TEST_F(TGEKModelTest, interpolate) {

	generate2DHimmelblauDataForTGEKModel(3);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);

	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.findIndicesOfDifferentiatedBasisFunctionLocations();
	rowvec x(2);
	x(0) = 0.3;
	x(1) = 0.25;

	double ftilde = testModel.interpolate(x);

}

TEST_F(TGEKModelTest, calculatePhiMatrix) {

	generate2DHimmelblauDataForTGEKModel(3);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
	//	testModel.setDisplayOn();
	testModel.readData();
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);

	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.setNumberOfDifferentiatedBasisFunctionsUsed(2);
	testModel.findIndicesOfDifferentiatedBasisFunctionLocations();
	testModel.generateWeightingMatrix();

	testModel.calculatePhiMatrix();

	bool ifResultsAreOk = testModel.checkPhiMatrix();

	ASSERT_TRUE(ifResultsAreOk);

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

	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);
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

	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);
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
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);
	testModel.setNumberOfTrainingIterations(1000);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.train();

	ASSERT_TRUE(testModel.ifModelTrainingIsDone);

}

TEST_F(TGEKModelTest, addNewSampleToData) {

	generate2DHimmelblauDataForTGEKModel(10);

	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
		//	testModel.setDisplayOn();
	testModel.readData();
	vec ub(2);
	ub.fill(6.0);
	vec lb(2);
	lb.fill(-6.0);
	testModel.setBoxConstraints(lb, ub);

	rowvec newSamplePoint(6, fill::randu);
	newSamplePoint(2) = 1.2;


	testModel.addNewSampleToData(newSamplePoint);

	mat data = testModel.getRawData();

	ASSERT_TRUE(data.n_rows == 11);

}



//TEST_F(TGEKModelTest, calculateOutSampleError1DFunction) {
//
//	generate1DTestFunctionDataForTGEKModel(10,2000);
//	trainingData.print();
//	testData.print();
//
//
//	testModel.setNameOfInputFile("trainingSamplesHimmelblauTGEK.csv");
//	//	testModel.setDisplayOn();
//	testModel.readData();
//	vec ub(1); ub.fill(6.0);
//	vec lb(1); lb.fill(0.0);
//	testModel.setBoxConstraints(lb, ub);
//	testModel.setNumberOfTrainingIterations(1000);
//	testModel.normalizeData();
//	testModel.initializeSurrogateModel();
//	testModel.setDisplayOn();
//	testModel.train();
//
//	testModel.printHyperParameters();
//
//
//	testModel.setNameOfInputFileTest("testSamplesHimmelblauTGEK.csv");
//	testModel.readDataTest();
//	testModel.normalizeDataTest();
//
//
//
//	double error = testModel.calculateOutSampleError();
//
//	printScalar(error);
//
//	testModel.saveTestResults();
//
//
//
//
//
//}

