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

#include "test_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef TEST_TESTFUNCTIONS

TEST(testTestFunctions, GenerateSamplesInput){

	TestFunction testFun("testFunction",2);
	testFun.setBoxConstraints(0.0, 2.0);
	testFun.numberOfTrainingSamples = 100;
	testFun.numberOfTestSamples     = 10;
	testFun.generateSamplesInputTrainingData();
	testFun.generateSamplesInputTestData();

	mat samplesInput = testFun.trainingSamplesInput;


	ASSERT_EQ(samplesInput.n_rows, 100);
	ASSERT_EQ(samplesInput.n_cols, 2);


	samplesInput = testFun.testSamplesInput;

	ASSERT_EQ(samplesInput.n_rows, 10);
	ASSERT_EQ(samplesInput.n_cols, 2);

}





TEST(testTestFunctions, evaluate){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr = LinearTF1;

	rowvec dv(2); dv(0) = 1.0; dv(1) = 2.0;
	Design testDesign(dv);

	testFun.evaluationSelect = 1;
	testFun.evaluate(testDesign);
	double expectedResult = 2*dv(0)+3*dv(1)+1.5;

	EXPECT_EQ(testDesign.trueValue, expectedResult);

}

TEST(testTestFunctions, evaluateAdjoint){

	TestFunction testFun("testFunction",2);
	testFun.adj_ptr = LinearTF1Adj;

	rowvec dv(2); dv(0) = 1.0; dv(1) = 2.0;
	Design testDesign(dv);

	testFun.evaluationSelect = 2;
	testFun.evaluate(testDesign);
	double expectedResult = 2*dv(0)+3*dv(1)+1.5;

	EXPECT_EQ(testDesign.trueValue, expectedResult);
	EXPECT_EQ(testDesign.gradient(0), 2.0);
	EXPECT_EQ(testDesign.gradient(1), 3.0);

}


TEST(testTestFunctions, evaluateTangent){

	TestFunction testFun("testFunction",2);
	testFun.tan_ptr = LinearTF1Tangent;

	rowvec dv(2); dv(0) = 1.0; dv(1) = 2.0;
	rowvec d(2);  d(0) =  1.0;  d(1) = 1.0;
	Design testDesign(dv);
	testDesign.tangentDirection = d;

	testFun.evaluationSelect = 3;
	testFun.evaluate(testDesign);
	double expectedResult = 5.0;

	EXPECT_EQ(testDesign.tangentValue, expectedResult );

	expectedResult = 2*dv(0)+3*dv(1)+1.5;
	EXPECT_EQ(testDesign.trueValue, expectedResult);



}

TEST(testTestFunctions, generateTestSamples){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr = LinearTF1;
	testFun.filenameTestData = "LinearTF1Test.csv";
	testFun.numberOfTestSamples = 20;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTestSamples();

	mat testData = testFun.testSamples;
	EXPECT_EQ(testData.n_cols, 3 );
	EXPECT_EQ(testData.n_rows, 20 );

	mat testDataRead;
	testDataRead.load("LinearTF1Test.csv", csv_ascii);

	EXPECT_TRUE(isEqual(testDataRead,testData, 10E-8));

	remove("LinearTF1Test.csv");
}




TEST(testTestFunctions, generateTrainingSamples){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr = LinearTF1;
	testFun.filenameTrainingData = "LinearTF1.csv";
	testFun.numberOfTrainingSamples = 20;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamples();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 3 );
	EXPECT_EQ(trainingData.n_rows, 20 );


	mat trainingDataRead;
	trainingDataRead.load("LinearTF1.csv", csv_ascii);

	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	remove("LinearTF1.csv");
}

TEST(testTestFunctions, generateTrainingSamplesAdjoint){

	TestFunction testFun("testFunction",2);
	testFun.adj_ptr = LinearTF1Adj;
	testFun.filenameTrainingData = "LinearTF1.csv";
	testFun.numberOfTrainingSamples = 20;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesWithAdjoints();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 5 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1.csv", csv_ascii);

	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	remove("LinearTF1.csv");
}

TEST(testTestFunctions, generateTrainingSamplesTangent){

	TestFunction testFun("testFunction",2);
	testFun.tan_ptr = LinearTF1Tangent;
	testFun.filenameTrainingData = "LinearTF1.csv";
	testFun.numberOfTrainingSamples = 20;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesWithTangents();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 6 );
	EXPECT_EQ(trainingData.n_rows, 20 );


	mat trainingDataRead;
	trainingDataRead.load("LinearTF1.csv", csv_ascii);

	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	remove("LinearTF1.csv");
}

TEST(testTestFunctions, generateTrainingSamplesMLOnlyWithFunctionalValues){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr      = LinearTF1;
	testFun.func_ptrLowFi = LinearTF1LowFidelity;
	testFun.filenameTrainingDataHighFidelity = "LinearTF1HiFi.csv";
	testFun.filenameTrainingDataLowFidelity  = "LinearTF1LowFi.csv";
	testFun.numberOfTrainingSamples = 20;
	testFun.numberOfTrainingSamplesLowFi = 50;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesMultiFidelity();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 3 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1HiFi.csv", csv_ascii);
	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	mat trainingDataLowFi = testFun.trainingSamplesLowFidelity;


	trainingDataRead.load("LinearTF1LowFi.csv", csv_ascii);
	EXPECT_EQ(trainingDataLowFi.n_cols, 3 );
	EXPECT_EQ(trainingDataLowFi.n_rows, 50 );
	EXPECT_TRUE(isEqual(trainingDataRead,trainingDataLowFi, 10E-8));

	remove("LinearTF1HiFi.csv");
	remove("LinearTF1LowFi.csv");
}


TEST(testTestFunctions, generateTrainingSamplesMLOnlyWithAdjoints){

	TestFunction testFun("testFunction",2);
	testFun.adj_ptr      = LinearTF1Adj;
	testFun.adj_ptrLowFi = LinearTF1LowFidelityAdj;
	testFun.filenameTrainingDataHighFidelity = "LinearTF1HiFi.csv";
	testFun.filenameTrainingDataLowFidelity  = "LinearTF1LowFi.csv";
	testFun.numberOfTrainingSamples = 20;
	testFun.numberOfTrainingSamplesLowFi = 50;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesMultiFidelityWithAdjoint();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 5 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1HiFi.csv", csv_ascii);
	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	mat trainingDataLowFi = testFun.trainingSamplesLowFidelity;


	trainingDataRead.load("LinearTF1LowFi.csv", csv_ascii);
	EXPECT_EQ(trainingDataLowFi.n_cols, 5 );
	EXPECT_EQ(trainingDataLowFi.n_rows, 50 );
	EXPECT_TRUE(isEqual(trainingDataRead,trainingDataLowFi, 10E-8));

	remove("LinearTF1HiFi.csv");
	remove("LinearTF1LowFi.csv");
}

TEST(testTestFunctions, generateTrainingSamplesMLOnlyWithTangents){

	TestFunction testFun("testFunction",2);
	testFun.tan_ptr      = LinearTF1Tangent;
	testFun.tan_ptrLowFi = LinearTF1LowFidelityTangent;
	testFun.filenameTrainingDataHighFidelity = "LinearTF1HiFi.csv";
	testFun.filenameTrainingDataLowFidelity  = "LinearTF1LowFi.csv";
	testFun.numberOfTrainingSamples = 20;
	testFun.numberOfTrainingSamplesLowFi = 50;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesMultiFidelityWithTangents();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 6 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1HiFi.csv", csv_ascii);
	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	mat trainingDataLowFi = testFun.trainingSamplesLowFidelity;


	trainingDataRead.load("LinearTF1LowFi.csv", csv_ascii);
	EXPECT_EQ(trainingDataLowFi.n_cols, 6 );
	EXPECT_EQ(trainingDataLowFi.n_rows, 50 );
	EXPECT_TRUE(isEqual(trainingDataRead,trainingDataLowFi, 10E-8));

	remove("LinearTF1HiFi.csv");
	remove("LinearTF1LowFi.csv");
}



TEST(testTestFunctions, generateTrainingSamplesMLWithLowFiAdjoints){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr      = LinearTF1;
	testFun.adj_ptrLowFi  = LinearTF1LowFidelityAdj;
	testFun.filenameTrainingDataHighFidelity = "LinearTF1HiFi.csv";
	testFun.filenameTrainingDataLowFidelity  = "LinearTF1LowFi.csv";
	testFun.numberOfTrainingSamples = 20;
	testFun.numberOfTrainingSamplesLowFi = 50;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 3 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1HiFi.csv", csv_ascii);
	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	mat trainingDataLowFi = testFun.trainingSamplesLowFidelity;


	trainingDataRead.load("LinearTF1LowFi.csv", csv_ascii);
	EXPECT_EQ(trainingDataLowFi.n_cols, 5 );
	EXPECT_EQ(trainingDataLowFi.n_rows, 50 );
	EXPECT_TRUE(isEqual(trainingDataRead,trainingDataLowFi, 10E-8));

	remove("LinearTF1HiFi.csv");
	remove("LinearTF1LowFi.csv");
}

TEST(testTestFunctions, generateTrainingSamplesMLWithLowFiTangents){

	TestFunction testFun("testFunction",2);
	testFun.func_ptr      = LinearTF1;
	testFun.tan_ptrLowFi  = LinearTF1LowFidelityTangent;
	testFun.filenameTrainingDataHighFidelity = "LinearTF1HiFi.csv";
	testFun.filenameTrainingDataLowFidelity  = "LinearTF1LowFi.csv";
	testFun.numberOfTrainingSamples = 20;
	testFun.numberOfTrainingSamplesLowFi = 50;

	testFun.setBoxConstraints(-4, 8);
	testFun.generateTrainingSamplesMultiFidelityWithLowFiTangents();

	mat trainingData = testFun.trainingSamples;
	EXPECT_EQ(trainingData.n_cols, 3 );
	EXPECT_EQ(trainingData.n_rows, 20 );

	mat trainingDataRead;
	trainingDataRead.load("LinearTF1HiFi.csv", csv_ascii);
	EXPECT_TRUE(isEqual(trainingDataRead,trainingData, 10E-8));

	mat trainingDataLowFi = testFun.trainingSamplesLowFidelity;


	trainingDataRead.load("LinearTF1LowFi.csv", csv_ascii);
	EXPECT_EQ(trainingDataLowFi.n_cols, 6 );
	EXPECT_EQ(trainingDataLowFi.n_rows, 50 );
	EXPECT_TRUE(isEqual(trainingDataRead,trainingDataLowFi, 10E-8));

	remove("LinearTF1HiFi.csv");
	remove("LinearTF1LowFi.csv");
}
#endif

