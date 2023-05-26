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

#include "multi_level_method.hpp"
#include "matrix_vector_operations.hpp"
#include "standard_test_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef MULTILEVEL_MODEL_TEST

class MultiLevelModelTest : public ::testing::Test {
protected:
	void SetUp() override {


		lb = zeros<vec>(2);
		ub = zeros<vec>(2);
		lb.fill(-6.0);
		ub.fill(6.0);

		boxConstraints.setBounds(lb,ub);

		testModel.setDimension(2);

		himmelblauFunction.function.filenameTrainingDataHighFidelity = "himmelblauHiFi.csv";
		himmelblauFunction.function.filenameTrainingDataLowFidelity = "himmelblauLowFi.csv";
		himmelblauFunction.function.filenameTestData = "himmelblauTest.csv";
		himmelblauFunction.function.numberOfTrainingSamples = 50;
		himmelblauFunction.function.numberOfTrainingSamplesLowFi = 100;


	}

	void TearDown() override {}


	MultiLevelModel testModel;
	vec lb;
	vec ub;
	HimmelblauFunction himmelblauFunction;

	Bounds boxConstraints;

};


TEST_F(MultiLevelModelTest, determineAlpha){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	testModel.determineAlpha();

	double alpha = testModel.getAlpha();

	double diff = fabs(1.0 - alpha);

	ASSERT_TRUE(diff>10E-10);


}






TEST_F(MultiLevelModelTest, constructor) {


	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifErrorDataIsSet);
	ASSERT_FALSE(testModel.ifInitialized);

}

TEST_F(MultiLevelModelTest, bindModels){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();

	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");

	ASSERT_TRUE(testModel.ifSurrogateModelsAreSet);

}

TEST_F(MultiLevelModelTest, bindModelsWithAdjointLowFi){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();

	testModel.setIDLowFiModel(GRADIENT_ENHANCED);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");

	ASSERT_TRUE(testModel.ifSurrogateModelsAreSet);


}




TEST_F(MultiLevelModelTest, setBoxConstraints){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();

	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);

	ASSERT_TRUE(testModel.ifBoxConstraintsAreSet);



}


TEST_F(MultiLevelModelTest, readData){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();

	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.readData();

	ASSERT_TRUE(testModel.ifDataIsRead);

}

TEST_F(MultiLevelModelTest, readDataWithLowFiTangents){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiTangents();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");


	testModel.setIDLowFiModel(TANGENT);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");
//	testModel.setDisplayOn();
	testModel.readData();

	ASSERT_TRUE(testModel.ifDataIsRead);

}

TEST_F(MultiLevelModelTest, normalizeData){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiTangents();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");


	testModel.setIDLowFiModel(TANGENT);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
//	testModel.setDisplayOn();
	testModel.readData();
	testModel.prepareAndReadErrorData();
	testModel.normalizeData();

	ASSERT_TRUE(testModel.ifDataIsRead);

}





TEST_F(MultiLevelModelTest, testfindIndexHiFiToLowFiData){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();

	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);
	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.readData();

	unsigned int index = testModel.findIndexHiFiToLowFiData(2);


	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();


	rowvec dx = dataLowFi.row(index) - dataHiFi.row(2);

	for(unsigned int i=0; i<2; i++){

		EXPECT_LT(dx(i),10E-10);
	}

}





TEST_F(MultiLevelModelTest, initializeSurrogateModel){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
//	testModel.setDisplayOn();
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();
	testModel.initializeSurrogateModel();


	ASSERT_TRUE(testModel.ifInitialized);

}



TEST_F(MultiLevelModelTest, trainLowFidelityModel){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();


	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();
	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainLowFidelityModel();

	mat samplesLowFi;
	samplesLowFi.load("himmelblauLowFi.csv",csv_ascii);


	double SE = 0.0;
	for(int i=0; i<testModel.getNumberOfLowFiSamples(); i++){

		rowvec x = samplesLowFi.row(i);
		rowvec xIn(2);
		xIn(0) = x(0) + 0.01;
		xIn(1) = x(1) + 0.01;

		double f = HimmelblauLowFi(xIn.memptr());
		xIn = normalizeRowVector(xIn, lb, ub);

		double ftilde = testModel.interpolateLowFi(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/testModel.getNumberOfLowFiSamples();

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 200.0);

}





TEST_F(MultiLevelModelTest, trainErrorModel){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainErrorModel();

	mat samplesError;
	samplesError.load("himmelblauModel_Error.csv",csv_ascii);


	double alpha = testModel.getAlpha();
	double squaredError = 0.0;
	for(int i=0; i<testModel.getNumberOfHiFiSamples(); i++){

		rowvec x = samplesError.row(i);
		double xp[2];
		xp[0] = x(0)+0.01; xp[1] = x(1)+0.01;
		rowvec xIn(2);
		xIn(0) = x(0)+0.01;
		xIn(1) = x(1)+0.01;

		xIn = normalizeRowVector(xIn, lb, ub);
		double ftilde = testModel.interpolateError(xIn);
		double f = Himmelblau(xp) - alpha*HimmelblauLowFi(xp);
		squaredError+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"ftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n\n";
#endif


	}


	double MSE = squaredError/testModel.getNumberOfHiFiSamples();



#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);


}


TEST_F(MultiLevelModelTest, testInterpolate){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);

	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(1000);


	testModel.train();

	double SE = 0.0;
	for(int i=0; i<100; i++){

		rowvec x(2);
		x = generateRandomRowVector(lb, ub);
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



TEST_F(MultiLevelModelTest, testInterpolateWithVariance){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(500);

	testModel.train();

	mat XLowFi = testModel.getRawDataLowFidelity();
	mat XHiFi = testModel.getRawDataHighFidelity();

	unsigned int someIndex = 3;
	unsigned int index = testModel.findIndexHiFiToLowFiData(someIndex);

	rowvec sampleLF = XLowFi.row(index);
	rowvec sampleHF = XHiFi.row(someIndex);

	//	sampleLF.print();
	//	sampleHF.print();

	double f = sampleHF(2);

	rowvec xp(2); xp(0) = sampleHF(0); xp(1) = sampleHF(1);

	xp = normalizeRowVector(xp, lb, ub);

	double ftilde = 0.0;
	double ssqr   = 10.0;

	testModel.interpolateWithVariance(xp, &ftilde, &ssqr);

	//	printScalar(ftilde);
	//	printScalar(f);
	//	printScalar(ssqr);

	double errorInf = fabs(f-ftilde);

	EXPECT_LT(errorInf,10E-06);
	EXPECT_LT(ssqr,10E-06);


}




TEST_F(MultiLevelModelTest, testAddNewSampleToData){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();


	rowvec newLowFiSample(3);
	vec x(2);
	x(0) = generateRandomDouble(lb(0),ub(0) );
	x(1) = generateRandomDouble(lb(1),ub(1) );

	double f = HimmelblauLowFi(x);
	newLowFiSample(0) = x(0);
	newLowFiSample(1) = x(1);
	newLowFiSample(2) = f;

	testModel.addNewLowFidelitySampleToData(newLowFiSample);

	ASSERT_EQ(testModel.getNumberOfLowFiSamples(),101);


}

TEST_F(MultiLevelModelTest, testAddNewHiFiSampleToData){

	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

	testModel.setinputFileNameHighFidelityData("himmelblauHiFi.csv");
	testModel.setinputFileNameLowFidelityData("himmelblauLowFi.csv");
	//	testModel.setDisplayOn();


	testModel.setIDLowFiModel(ORDINARY_KRIGING);
	testModel.setIDHiFiModel(ORDINARY_KRIGING);


	testModel.bindModels();
	testModel.setName("himmelblauModel");
	testModel.setBoxConstraints(boxConstraints);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	rowvec newHiFiSample(3);
	rowvec newLowFiSample(3);
	vec x(2);
	x(0) = generateRandomDouble(lb(0),ub(0) );
	x(1) = generateRandomDouble(lb(1),ub(1) );

	double f = HimmelblauLowFi(x);
	newLowFiSample(0) = x(0);
	newLowFiSample(1) = x(1);
	newLowFiSample(2) = f;

	f = Himmelblau(x);
	newHiFiSample(0) = x(0);
	newHiFiSample(1) = x(1);
	newHiFiSample(2) = f;


	testModel.addNewLowFidelitySampleToData(newLowFiSample);
	testModel.addNewSampleToData(newHiFiSample);

	ASSERT_EQ(testModel.getNumberOfHiFiSamples(),51);


}



#endif


