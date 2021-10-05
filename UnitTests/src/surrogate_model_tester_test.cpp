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

#include<gtest/gtest.h>
#include "surrogate_model_tester.hpp"
#include "matrix_vector_operations.hpp"
#include "bounds.hpp"
#include "random_functions.hpp"
#include "test_functions.hpp"


#define TEST_SURROGATE_MODEL_TESTER
#ifdef TEST_SURROGATE_MODEL_TESTER

TEST(testSurrogateModelTester, testConstructor){

	SurrogateModelTester surrogateTester;

	unsigned int dimGet = surrogateTester.getDimension();
	ASSERT_EQ(dimGet,0);

}

TEST(testSurrogateModelTester, testsetDimension){

	SurrogateModelTester surrogateTester;

	surrogateTester.setDimension(2);
	unsigned int dimGet = surrogateTester.getDimension();
	ASSERT_EQ(dimGet,2);

}

TEST(testSurrogateModelTester, testsetSurrogateModel){

	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingData("trainingData.csv");
	surrogateTester.setFileNameTestData("test.csv");
	surrogateTester.setSurrogateModel(LINEAR_REGRESSION);

	bool ifSurrogateModelSpecified = surrogateTester.isSurrogateModelSpecified();
	ASSERT_EQ(ifSurrogateModelSpecified,true);




}


TEST(testSurrogateModelTester, testsetBoxConstraints){

	unsigned int dim = 5;
	vec lowerBounds(dim, fill::zeros);
	vec upperBounds(dim, fill::ones);
	Bounds boxConstraints(lowerBounds , upperBounds);

	SurrogateModelTester surrogateTester;
	surrogateTester.setBoxConstraints(boxConstraints);

	Bounds boxConstraintsGet = surrogateTester.getBoxConstraints();

	unsigned int dimBoxConstraints = boxConstraintsGet.getDimension();
	ASSERT_EQ(dimBoxConstraints,5);

}


TEST(testSurrogateModelTester, testperformSurrogateModelTestLinearRegression){

	unsigned int dim = 3;
	unsigned int N = 50;

	mat dataMatrix(N,dim+1, fill::randu);
	mat dataMatrixTest(N,dim, fill::randu);

	for(unsigned int i=0; i<N; i++){

		dataMatrix(i,dim) = 2.0*dataMatrix(i,0) + 2.0*dataMatrix(i,1) - dataMatrix(i,2) + 0.01*generateRandomDouble(-1.0,1.0);

	}

	saveMatToCVSFile(dataMatrix,"trainingData.csv");
	saveMatToCVSFile(dataMatrixTest,"testData.csv");

	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingData("trainingData.csv");
	surrogateTester.setFileNameTestData("testData.csv");

	surrogateTester.setSurrogateModel(LINEAR_REGRESSION);

	//	surrogateTester.setDisplayOn();

	surrogateTester.performSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);


	vec yTilde = results.col(dim);
	double squaredError = 0.0;
	for(unsigned int i=0; i<N; i++){

		double yExact =  2.0*dataMatrixTest(i,0) + 2.0*dataMatrixTest(i,1) - dataMatrixTest(i,2);
		squaredError+= (yTilde(i) - yExact) * (yTilde(i) - yExact);
#if 0
		std::cout<<"yExact = "<<yExact<<" yTilde = "<<yTilde(i)<<"\n";
#endif
	}

	double meanSquaredError =  squaredError/N;
#if 0
	std::cout<<"MSE = "<<meanSquaredError<<"\n";
#endif

	EXPECT_LT(meanSquaredError, 0.001);

	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testData.csv");



}

TEST(testSurrogateModelTester, testperformSurrogateModelTestOrdinaryKriging){

	unsigned int dim = 2;
	unsigned int N = 100;

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(Himmelblau);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);

	himmelblauFunction.setNumberOfTrainingSamples(N);
	himmelblauFunction.generateSamplesInputTrainingData();
	himmelblauFunction.generateTrainingSamples();
	mat trainingData = himmelblauFunction.getTrainingSamples();

	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();


	saveMatToCVSFile(trainingData,"trainingData.csv");
	saveMatToCVSFile(testDataInput,"testDataInput.csv");

	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingData("trainingData.csv");
	surrogateTester.setFileNameTestData("testDataInput.csv");
	surrogateTester.setNumberOfTrainingIterations(1000);


	surrogateTester.setSurrogateModel(ORDINARY_KRIGING);
	//	surrogateTester.setDisplayOn();


	surrogateTester.performSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	vec yTilde = results.col(dim);

	double squaredError = 0.0;
	for(unsigned int i=0; i<N; i++){

		double yExact =  testData(i,dim);
		squaredError+= (yTilde(i) - yExact) * (yTilde(i) - yExact);
#if 0
		std::cout<<"yExact = "<<yExact<<" yTilde = "<<yTilde(i)<<"\n";
#endif
	}

	double meanSquaredError =  squaredError/N;
#if 0
	std::cout<<"MSE = "<<meanSquaredError<<"\n";
#endif

	EXPECT_LT(meanSquaredError, 500.0);


	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testDataInput.csv");

}

TEST(testSurrogateModelTester, testperformSurrogateModelTestUniversalKriging){


	unsigned int dim = 2;
	unsigned int N = 100;

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(Himmelblau);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);

	himmelblauFunction.setNumberOfTrainingSamples(N);
	himmelblauFunction.generateSamplesInputTrainingData();
	himmelblauFunction.generateTrainingSamples();
	mat trainingData = himmelblauFunction.getTrainingSamples();

	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();


	saveMatToCVSFile(trainingData,"trainingData.csv");
	saveMatToCVSFile(testDataInput,"testDataInput.csv");

	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingData("trainingData.csv");
	surrogateTester.setFileNameTestData("testDataInput.csv");
	surrogateTester.setNumberOfTrainingIterations(1000);


	surrogateTester.setSurrogateModel(UNIVERSAL_KRIGING);
	//	surrogateTester.setDisplayOn();

	surrogateTester.performSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	vec yTilde = results.col(dim);

	double squaredError = 0.0;
	for(unsigned int i=0; i<N; i++){

		double yExact =  testData(i,dim);
		squaredError+= (yTilde(i) - yExact) * (yTilde(i) - yExact);
#if 0
		std::cout<<"yExact = "<<yExact<<" yTilde = "<<yTilde(i)<<"\n";
#endif
	}

	double meanSquaredError =  squaredError/N;
#if 0
	std::cout<<"MSE = "<<meanSquaredError<<"\n";
#endif

	EXPECT_LT(meanSquaredError, 1000.0);


	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testDataInput.csv");

}

TEST(testSurrogateModelTester, testperformSurrogateModelTestAggregation){


	unsigned int dim = 2;
	unsigned int N = 100;

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(HimmelblauAdj);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);

	himmelblauFunction.setNumberOfTrainingSamples(N);
	himmelblauFunction.generateSamplesInputTrainingData();
	himmelblauFunction.generateTrainingSamples();
	mat trainingData = himmelblauFunction.getTrainingSamples();


	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();

	saveMatToCVSFile(trainingData,"trainingData.csv");
	saveMatToCVSFile(testDataInput,"testDataInput.csv");


	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingData("trainingData.csv");
	surrogateTester.setFileNameTestData("testDataInput.csv");
	surrogateTester.setNumberOfTrainingIterations(1000);


	surrogateTester.setSurrogateModel(AGGREGATION);

//	surrogateTester.setDisplayOn();

	surrogateTester.performSurrogateModelTest();


	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	vec yTilde = results.col(dim);

	double squaredError = 0.0;
	for(unsigned int i=0; i<N; i++){

		double yExact =  testData(i,dim);
		squaredError+= (yTilde(i) - yExact) * (yTilde(i) - yExact);
#if 0
		std::cout<<"yExact = "<<yExact<<" yTilde = "<<yTilde(i)<<"\n";
#endif
	}

	double meanSquaredError =  squaredError/N;
#if 0
	std::cout<<"MSE = "<<meanSquaredError<<"\n";
#endif

	EXPECT_LT(meanSquaredError, 10000.0);



	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testDataInput.csv");


}

TEST(testSurrogateModelTester, testperformSurrogateModelTestMultiLevel){


	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;

	unsigned int N = 100;
	unsigned int dim = 2;

	generateHimmelblauDataMultiFidelity("highFidelityTrainingData.csv","lowFidelityTraingData.csv",nSamplesHiFi ,nSamplesLowFi);

	SurrogateModelTester surrogateTester;
	surrogateTester.setName("testModel");
	surrogateTester.setFileNameTrainingDataLowFidelity("lowFidelityTraingData.csv");
	surrogateTester.setFileNameTrainingData("highFidelityTrainingData.csv");


	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(HimmelblauAdj);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);



	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();

	saveMatToCVSFile(testDataInput,"testDataInput.csv");


	surrogateTester.setFileNameTestData("testDataInput.csv");

	surrogateTester.setNumberOfTrainingIterations(1000);


	surrogateTester.setSurrogateModel(MULTI_LEVEL);

//	surrogateTester.setDisplayOn();

	surrogateTester.performSurrogateModelTest();


	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	vec yTilde = results.col(dim);

	double squaredError = 0.0;
	for(unsigned int i=0; i<N; i++){

		double yExact =  testData(i,dim);
		squaredError+= (yTilde(i) - yExact) * (yTilde(i) - yExact);
#if 0
		std::cout<<"yExact = "<<yExact<<" yTilde = "<<yTilde(i)<<"\n";
#endif
	}

	double meanSquaredError =  squaredError/N;
#if 0
	std::cout<<"MSE = "<<meanSquaredError<<"\n";
#endif

	EXPECT_LT(meanSquaredError, 10000.0);



	remove("surrogateTest.csv");
	remove("lowFidelityTraingData.csv");
	remove("highFidelityTrainingData.csv");
	remove("testDataInput.csv");
}



#endif
