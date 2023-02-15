/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>





class KrigingModelTest : public ::testing::Test {
protected:
	void SetUp() override {


		unsigned int N = 200;
		generate2DEggholderDataForKrigingModel(N);
		testModel2D.setName("Eggholder");
		testModel2D.setNameOfInputFile("Eggholder.csv");
		testModel2D.readData();
		testModel2D.setBoxConstraints(0.0, 200.0);
		testModel2D.normalizeData();
		testModel2D.setNameOfInputFileTest("EggholderTest.csv");
		testModel2D.readDataTest();
		testModel2D.normalizeDataTest();


		testModel2D.initializeSurrogateModel();

		N = 10;
		mat testData1D(N,2);
		testData1D(0,0) = 0.0;
		testData1D(1,0) = 0.1;
		testData1D(2,0) = 0.15;
		testData1D(3,0) = 0.2;
		testData1D(4,0) = 0.25;
		testData1D(5,0) = 0.4;
		testData1D(6,0) = 0.5;
		testData1D(7,0) = 0.75;
		testData1D(8,0) = 0.85;
		testData1D(9,0) = 1.0;

		testData1D(0,1) = 3.027209981231713;
		testData1D(1,1) = -0.656576774305574;
		testData1D(2,1) = -0.978280648621704;
		testData1D(3,1) =  -0.639727105946563;
		testData1D(4,1) =  -0.210367746201974;
		testData1D(5,1) =   0.114776974543924;
		testData1D(6,1) =   0.909297426825682;
		testData1D(7,1) =  -5.993276716644615;
		testData1D(8,1) =  -0.798489161076149;
		testData1D(9,1) =  15.829731945974109;

		saveMatToCVSFile(testData1D, "testdata1D.csv");

		testModel1D.setName("testdata1D");
		testModel1D.setNameOfInputFile("testdata1D.csv");
		testModel1D.readData();
		testModel1D.setBoxConstraints(0.0, 1.0);
		testModel1D.normalizeData();
		testModel1D.initializeSurrogateModel();


	}

	void TearDown() override {

		remove("testdata1D.csv");
		remove("Eggholder.csv");
		remove("EggholderTest.csv");
		remove("LinearTF.csv");
		remove("LinearTFTest.csv");


	}


	void generate2DEggholderDataForKrigingModel(unsigned int N){

		TestFunction testFunctionEggholder("Eggholder",2);

		testFunctionEggholder.func_ptr = Eggholder;
		testFunctionEggholder.setBoxConstraints(0.0,200.0);
		testFunctionEggholder.numberOfTrainingSamples = N;
		testFunctionEggholder.filenameTrainingData = "Eggholder.csv";
		testFunctionEggholder.numberOfTestSamples = N;
		testFunctionEggholder.filenameTestData = "EggholderTest.csv";

		testFunctionEggholder.generateTrainingSamples();
		testFunctionEggholder.generateTestSamples();

		trainingDataEggholder = testFunctionEggholder.trainingSamples;
		testDataEggholder = testFunctionEggholder.testSamples;


	}

	void generate2DLinearTestFunctionDataForKrigingModel(unsigned int N){

		TestFunction testFunctionLinear("LinearTF1",2);

		testFunctionLinear.func_ptr =  LinearTF1;
		testFunctionLinear.setBoxConstraints(-10.0,10.0);
		testFunctionLinear.numberOfTrainingSamples = N;
		testFunctionLinear.filenameTrainingData = "LinearTF.csv";
		testFunctionLinear.numberOfTestSamples = N;
		testFunctionLinear.filenameTestData = "LinearTFTest.csv";
		trainingDataEggholder = testFunctionLinear.trainingSamples;
		testDataEggholder = testFunctionLinear.testSamples;


	}



	KrigingModel testModel2D;
	KrigingModel testModel2DLinear;
	KrigingModel testModel1D;
	KrigingHyperParameterOptimizer testOptimizer;
	mat trainingDataEggholder;
	mat testDataEggholder;
	mat trainingDataLinearTestFunction;
	mat testDataLinearTestFunction;
};


TEST_F(KrigingModelTest, updateAuxilliaryFields) {

	vec hyperParameters(4);
	hyperParameters(0) = 10.0;
	hyperParameters(1) = 10.0;
	hyperParameters(2) = 2.0;
	hyperParameters(3) = 2.0;

	testModel2D.setHyperParameters(hyperParameters);

	testModel2D.updateAuxilliaryFields();

	testModel2D.tryOnTestData();

	mat results;
	results.load("surrogateTestResults.csv", csv_ascii);

	vec squaredError = results.col(4);

	double normL2error = norm(squaredError,1);
	normL2error = normL2error/squaredError.size();

	EXPECT_LT(normL2error, 10E4);

	remove("surrogateTestResults.csv");
}



TEST_F(KrigingModelTest, updateModelWithNewData) {


	generate2DEggholderDataForKrigingModel(101);
	testModel2D.updateModelWithNewData();

	unsigned int N = testModel2D.getNumberOfSamples();

	ASSERT_TRUE(N == 101);

}

TEST_F(KrigingModelTest, testIfConstructorWorks) {

	ASSERT_TRUE(testModel2D.ifDataIsRead);
	ASSERT_TRUE(testModel2D.ifInitialized);
	ASSERT_TRUE(testModel2D.ifNormalized);
	ASSERT_TRUE(testModel2D.ifHasTestData);
	ASSERT_FALSE(testModel2D.areGradientsOn());

	ASSERT_TRUE(testModel2D.getDimension() == 2);
	std::string filenameDataInput = testModel2D.getNameOfInputFile();
	ASSERT_TRUE(filenameDataInput == "Eggholder.csv");


}



TEST_F(KrigingModelTest, calculateLikelihood) {

	vec hyperParameters(2);
	hyperParameters(0) = 5.0;
	hyperParameters(1) = 2.0;

	double L = testModel1D.calculateLikelihoodFunction(hyperParameters);

	double resultFromForresterCode = -24.202852114659308;
	double error = fabs(L - resultFromForresterCode);

	ASSERT_LT(error,0.01);


}


TEST_F(KrigingModelTest, inSampleErrorWithoutTraining) {

	vec xSample(1);
	xSample(0) = 0.25;
	testModel1D.setEpsilon(10E-6);
	testModel1D.updateAuxilliaryFields();

	double fTilde = testModel1D.interpolate(xSample);

	double resultFromForresterCode = -0.141583407517970;
	double error = fabs(fTilde - resultFromForresterCode);

	ASSERT_LT(error, 10E-6);


}




TEST_F(KrigingModelTest, testKrigingOptimizerinitializeKrigingModelObject) {

	KrigingHyperParameterOptimizer testOptimizer;

	testOptimizer.initializeKrigingModelObject(testModel2D);
	ASSERT_TRUE(testOptimizer.ifModelObjectIsSet);

}



TEST_F(KrigingModelTest, testKrigingOptimizertestKrigingOptimizerOptimize) {

	unsigned int dim = 2;

	vec hyperParameters(2*dim);
	hyperParameters(0) = 10.0;
	hyperParameters(dim-1) = 0.1;
	hyperParameters(dim) = 0.5;
	hyperParameters(dim+1) = 1.5;
	double L = testModel2D.calculateLikelihoodFunction(hyperParameters);

	KrigingHyperParameterOptimizer testOptimizer;
	testOptimizer.initializeKrigingModelObject(testModel2D);

	testOptimizer.setDimension(2*dim);
	Bounds boxConstraints(2*dim);
	vec lb(2*dim); lb(0) = 0.0; lb(1) = 0.0; lb(2) = 0.0; lb(3) = 0.0;
	vec ub(2*dim); ub(0) = 10.0; ub(1) = 10.0; ub(2) = 2.0; ub(3) = 2.0;
	boxConstraints.setBounds(lb,ub);
	testOptimizer.setBounds(boxConstraints);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(100*2*dim);
	testOptimizer.setNumberOfDeathsInAGeneration(100*dim);
	testOptimizer.setInitialPopulationSize(2*dim*100);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.setMaximumNumberOfGeneratedIndividuals(100000*2*dim);
	testOptimizer.setNumberOfGenerations(5);
	//	testOptimizer.setDisplayOn();


	testOptimizer.optimize();

	EAIndividual solution = testOptimizer.getSolution();

	vec optimizedHyperParameters = solution.getGenes();

	double Loptimized = testModel2D.calculateLikelihoodFunction(optimizedHyperParameters);

	ASSERT_TRUE(Loptimized>L);


}




TEST_F(KrigingModelTest, train) {


	unsigned int dim = 2;

	vec hyperParameters(2*dim);
	hyperParameters(0) = 10.0;
	hyperParameters(dim-1) = 0.1;
	hyperParameters(dim) = 0.5;
	hyperParameters(dim+1) = 1.5;
	double L = testModel2D.calculateLikelihoodFunction(hyperParameters);

	testModel2D.setNumberOfThreads(4);
	testModel2D.setWriteWarmStartFileOn("warmStartFile.csv");

	testModel2D.train();

	ASSERT_TRUE(testModel2D.ifModelTrainingIsDone);
	remove("warmStartFile.csv");

}

TEST_F(KrigingModelTest, trainWithWarmStart) {



	testModel2D.setNumberOfTrainingIterations(1000);
	testModel2D.setNumberOfThreads(4);
	testModel2D.setWriteWarmStartFileOn("warmStartFile.csv");
	testModel2D.train();
	testModel2D.setReadWarmStartFileOn("warmStartFile.csv");
	testModel2D.train();
	ASSERT_TRUE(testModel2D.ifModelTrainingIsDone);
	remove("warmStartFile.csv");
}



TEST_F(KrigingModelTest, calculateOutSampleError) {

	testModel2D.train();

	double errorInSample = testModel2D.calculateInSampleError();

	EXPECT_LT(errorInSample,10E-06);

	testModel2D.setNameOfInputFileTest("EggholderTest.csv");
	testModel2D.readDataTest();
	testModel2D.normalizeDataTest();
	testModel2D.setNameOfOutputFileTest("EggholderTestResults.csv");

	double errorOutSample = testModel2D.calculateOutSampleError();

	double RMSE = sqrt(errorOutSample);
	EXPECT_LT(RMSE,200.0);

}




TEST_F(KrigingModelTest, saveAndLoadHyperParameters) {

	testModel2D.setNumberOfTrainingIterations(100);
	testModel2D.train();
	//	testModel2D.printHyperParameters();

	testModel2D.saveHyperParameters();

	testModel2D.loadHyperParameters();


	//	testModel2D.printHyperParameters();

}




TEST_F(KrigingModelTest, linearModel) {

	generate2DLinearTestFunctionDataForKrigingModel(50);

	testModel2DLinear.setName("LinearTF");
	testModel2DLinear.setNameOfInputFile("LinearTF.csv");
	testModel2DLinear.readData();

	testModel2DLinear.setBoxConstraints(-10.0, 10.0);
	testModel2DLinear.normalizeData();
	testModel2DLinear.setLinearRegressionOn();
	testModel2DLinear.initializeSurrogateModel();

	testModel2DLinear.train();
	testModel2DLinear.setNameOfInputFileTest("LinearTFTest.csv");
	testModel2DLinear.readDataTest();
	testModel2DLinear.normalizeDataTest();


	double errorOutSample = testModel2DLinear.calculateOutSampleError();

	//	testModel2DLinear.printHyperParameters();

	double RMSE = sqrt(errorOutSample);
	EXPECT_LT(RMSE,0.001);

}






