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

#include "multi_level_method.hpp"
#include "matrix_vector_operations.hpp"
#include<gtest/gtest.h>
TEST(testMultiLevelModel, testConstructor){

	MultiLevelModel testModel("MLtestModel");
	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifBoundsAreSet);
	ASSERT_FALSE(testModel.ifInitialized);


}

TEST(testMultiLevelModel, testSetErrorModel){

	MultiLevelModel testModel("MLtestModel");



}


TEST(testMultiLevelModel, testfindIndexHiFiToLowFiData){

	mat testHifiData(10,9,fill::randu);
	mat testLofiData(20,9,fill::randu);

	for(unsigned int i=0; i<10; i++){

		testLofiData.row(i+10) = testHifiData.row(i);
	}

	testLofiData = shuffle(testLofiData);

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	unsigned int index = testModel.findIndexHiFiToLowFiData(2);


	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();


	rowvec dx = dataLowFi.row(index) - dataHiFi.row(2);

	for(unsigned int i=0; i<9; i++){

		EXPECT_LT(dx(i),10E-10);
	}


}

TEST(testMultiLevelModel, prepareErrorData){
	mat testHifiData(5,3,fill::randu);
	mat testLofiData(7,3,fill::randu);
	for(unsigned int i=0; i<5; i++){

		testLofiData.row(i+2) = testHifiData.row(i);
	}

	testLofiData(6,2) += 0.213;

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	testModel.prepareErrorData();

	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();
	mat dataError = testModel.getRawDataError();


	double difference = fabs( dataError(4,2) - (dataHiFi(4,2) -dataLowFi(6,2) ));

	EXPECT_LT(difference, 10E-08);
	ASSERT_TRUE(testModel.ifErrorDataIsSet);


}

TEST(testMultiLevelModel, prepareErrorDataWithShuffle){
	mat testHifiData(50,5,fill::randu);
	mat testLofiData(70,5,fill::randu);
	for(unsigned int i=0; i<50; i++){

		testLofiData.row(i+20) = testHifiData.row(i);
	}

	vec randomVec(70,fill::randu);
	testLofiData.col(4) = randomVec;
	testLofiData = shuffle(testLofiData);


	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	testModel.prepareErrorData();

	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();
	mat dataError = testModel.getRawDataError();

	unsigned int index = testModel.findIndexHiFiToLowFiData(2);


	double difference = fabs( dataError(2,4) - (dataHiFi(2,4) -dataLowFi(index,4) ));

	EXPECT_LT(difference, 10E-08);
	ASSERT_TRUE(testModel.ifErrorDataIsSet);


}




TEST(testMultiLevelModel, testReadData){
	mat testHifiData(100,9,fill::randu);
	mat testLofiData(200,9,fill::randu);

	for(unsigned int i=0; i<100; i++){

		testLofiData.row(i+100) = testHifiData.row(i);
	}

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");
	testModel.readData();
	ASSERT_TRUE(testModel.ifDataIsRead);



}

TEST(testMultiLevelModel, testSetParameterBounds){

	mat testHifiData(100,9,fill::randu);
	mat testLofiData(200,9,fill::randu);

	for(unsigned int i=0; i<100; i++){

		testLofiData.row(i+100) = testHifiData.row(i);
	}

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	vec lb(8); lb.fill(-2);
	vec ub(8); ub.fill(2);

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");

	testModel.readData();
	testModel.setParameterBounds(lb,ub);

}

TEST(testMultiLevelModel, testInterpolate){


	mat samplesHiFi(10,3);
	mat samplesLowFi(30,3);
	mat samplesInput(30,2);

	for (unsigned int i=0; i<samplesInput.n_rows; i++){

		rowvec x(2);
		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		samplesInput.row(i) = x;

	}


	/* we construct test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samplesLowFi.n_rows; i++){
		rowvec x(3);
		x(0) = samplesInput(i,0);
		x(1) = samplesInput(i,1);

		x(2) = x(0)*x(0) + x(1)*x(1);

		if(i<samplesHiFi.n_rows){

			samplesHiFi.row(i) = x;
		}
		x(2) += 0.01*generateRandomDouble(-1.0,1.0);
		samplesLowFi.row(i) = x;
	}


	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);
	samplesLowFi.save("LowFiData.csv", csv_ascii);
	samplesHiFi.save("HiFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LowFiData.csv");
	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.setParameterBounds(lb,ub);
	testModel.initializeSurrogateModel();



	rowvec xTest(2); xTest(0) = 0.5; xTest(1) = 0.5;
	rowvec xTestNorm = normalizeRowVector(xTest, lb, ub);

	double ftildeTest = testModel.interpolate(xTestNorm);

	double error = ftildeTest - 0.5;
	EXPECT_LT(error, 0.1);

}

TEST(testMultiLevelModel, testprapareTrainingDataForGammaOptimization){
	mat testHifiData(100,9,fill::randu);
	mat testLofiData(200,9,fill::randu);

	for(unsigned int i=0; i<100; i++){

		testLofiData.row(i+100) = testHifiData.row(i);
	}

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");
	testModel.readData();
	testModel.prepareTrainingDataForGammaOptimization();

	mat testData = testModel.getRawDataHighFidelityForGammaTest();
	mat trainingData = testModel.getRawDataHighFidelityForGammaTraining();

	rowvec x1 = testHifiData.row(5);
	rowvec x2 = testData.row(5);


	for(int i=0;i<9; i++){

		double error = fabs(x1(i) - x2(i));
		EXPECT_LT(error,10E-10);
	}

	x1 = testHifiData.row(25);
	x2 = trainingData.row(5);


	for(int i=0;i<9; i++){

		double error = fabs(x1(i) - x2(i));
		EXPECT_LT(error,10E-10);
	}



}

TEST(testMultiLevelModel, testTrainGamma){
	mat testHifiData(100,9,fill::randu);
	mat testLofiData(200,9,fill::randu);

	for(unsigned int i=0; i<100; i++){

		testLofiData.row(i+100) = testHifiData.row(i);
	}

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");
	testModel.readData();
	testModel.prepareTrainingDataForGammaOptimization();
	testModel.trainGamma();


}

