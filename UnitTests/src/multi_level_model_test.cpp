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
#include "test_functions.hpp"
#include<gtest/gtest.h>


TEST(testMultiLevelModel, testConstructor){

	MultiLevelModel testModel("MLtestModel");
	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);

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

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();


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
//	testModel.setDisplayOn();

	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();


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

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();


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

	vec lb(8);
	vec ub(8);

	lb.fill(0.0);
	ub.fill(1.0);

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");



	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();

	//	testModel.ifDisplay = true;

	testModel.setBoxConstraints(lb,ub);
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

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();

	testModel.setBoxConstraints(lb,ub);



}


TEST(testMultiLevelModel, testfindNearestNeighbourLowFidelity){

	unsigned int nSamplesLowFi = 200;
	unsigned int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel2");
//	testModel.setDisplayOn();

	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTestData.csv",csv_ascii);

	unsigned int someIndex = 14;
	rowvec x = samplesLowFi.row(someIndex);
	rowvec xp(2); xp(0) = x(0) + 0.001;  xp(1) = x(1)-0.001;

	xp = normalizeRowVector(xp, lb, ub);

	unsigned int indx = testModel.findNearestNeighbourLowFidelity(xp);

	EXPECT_EQ(indx, someIndex);


}

TEST(testMultiLevelModel, testfindNearestL1DistanceToALowFidelitySample){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel2");
	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTestData.csv",csv_ascii);

	rowvec x = samplesLowFi.row(14);
	rowvec xp(2); xp(0) = x(0);  xp(1) = x(1);

	xp = normalizeRowVector(xp, lb, ub);
	xp(0) += 0.0001;
	xp(1) -= 0.0001;

	double dist = testModel.findNearestL1DistanceToALowFidelitySample(xp);

#if 0
	std::cout<<"dist = "<<dist<<"\n";
#endif

	EXPECT_LT(fabs(dist-0.0002),10E-08 );


}



TEST(testMultiLevelModel, testfindNearestNeighbourHighFidelity){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel2");
	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	mat samples;
	samples.load("highFidelityTestData.csv",csv_ascii);

	rowvec x = samples.row(14);
	rowvec xp(2); xp(0) = x(0) + 0.001;  xp(1) = x(1)-0.001;

	xp = normalizeRowVector(xp, lb, ub);

	unsigned int indx = testModel.findNearestNeighbourHighFidelity(xp);

	EXPECT_EQ(indx, 14);


}


TEST(testMultiLevelModel, testtrainLowFidelityModel){


	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel2");
	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainLowFidelityModel();

	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTestData.csv",csv_ascii);


	double SE = 0.0;
	for(int i=0; i<nSamplesLowFi; i++){

		rowvec x = samplesLowFi.row(i);
		double xp[2]; xp[0] = x(0)+0.1; xp[1] = x(1)+0.1;
		rowvec xIn(2);
		xIn(0) = x(0) + 0.1;
		xIn(1) = x(1) + 0.1;

		double f = Himmelblau(xIn.memptr());
		xIn = normalizeRowVector(xIn, lb, ub);

		double ftilde = testModel.interpolateLowFi(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/nSamplesLowFi;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 200.0);




}



TEST(testMultiLevelModel, testtrainLowFidelityModelWithGradient){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;

	generateHimmelblauDataMultiFidelityWithGradients("highFidelityTestDataWithGradient.csv","lowFidelityTestDataWithGradient.csv",nSamplesHiFi ,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModelWithGradients");
//	testModel.setDisplayOn();

	testModel.setinputFileNameHighFidelityData("highFidelityTestDataWithGradient.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestDataWithGradient.csv");


	testModel.setGradientsOn();
	testModel.setGradientsOnLowFi();
	testModel.setGradientsOnHiFi();



	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainLowFidelityModel();


	double SE = 0.0;
	for(int i=0; i<100; i++){

		rowvec x(2);
		x = generateRandomRowVector(lb,ub);
		double xp[2]; xp[0] = x(0); xp[1] = x(1);
		rowvec xIn(2);
		xIn(0) = x(0);
		xIn(1) = x(1);

		double f = Himmelblau(xIn.memptr());
		xIn = normalizeRowVector(xIn, lb, ub);

		double ftilde = testModel.interpolateLowFi(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/100;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);


}

TEST(testMultiLevelModel, testtrainErrorModel){

	unsigned int numberOfHiFiSamples = 50;
	unsigned int numberOfLowFiSamples = 200;


	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",numberOfHiFiSamples,numberOfLowFiSamples);


	MultiLevelModel testModel("MLtestModel");

	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");


	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainErrorModel();

	mat samplesError;
	samplesError.load("MLtestModel_Error.csv",csv_ascii);


	double squaredError = 0.0;
	for(int i=0; i<numberOfHiFiSamples; i++){

		rowvec x = samplesError.row(i);
		double xp[2]; xp[0] = x(0)+0.1; xp[1] = x(1)+0.1;
		rowvec xIn(2);
		xIn(0) = x(0)+0.1;
		xIn(1) = x(1)+0.1;

		xIn = normalizeRowVector(xIn, lb, ub);
		double ftilde = testModel.interpolateError(xIn);
		double f = -Waves2D(xp)*10.0;
		squaredError+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"ftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n\n";
#endif


	}


	double MSE = squaredError/numberOfHiFiSamples;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);


}

TEST(testMultiLevelModel, testdetermineGammaBasedOnData){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();

	testModel.determineGammaBasedOnData();



}



TEST(testMultiLevelModel, testInterpolate){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;
	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	MultiLevelModel testModel("MLtestModel2");
	testModel.setinputFileNameHighFidelityData("highFidelityTestData.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTestData.csv");

	vec lb(2); lb.fill(-6.0);
	vec ub(2); ub.fill(6.0);

	testModel.setBoxConstraints(lb,ub);
	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(1000);

	testModel.train();

	double SE = 0.0;
	for(int i=0; i<100; i++){

		rowvec x(2);
		x = generateRandomRowVector(lb,ub);
		double xp[2]; xp[0] = x(0); xp[1] = x(1);
		rowvec xIn(2);
		xIn(0) = x(0);
		xIn(1) = x(1);

		double f = Himmelblau(xIn.memptr());
		xIn = normalizeRowVector(xIn, lb, ub);

		double ftilde = testModel.interpolate(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/100;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);



}




