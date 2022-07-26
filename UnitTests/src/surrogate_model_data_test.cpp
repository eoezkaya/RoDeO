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
#include "surrogate_model_data.hpp"
#include "matrix_vector_operations.hpp"
using std::string;

#define TEST_SURROGATE_MODEL_DATA
#ifdef TEST_SURROGATE_MODEL_DATA

class SurrogateModelDataTest : public ::testing::Test {
protected:
	void SetUp() override {


	}

	void TearDown() override {

		remove("testData.csv");

	}

	SurrogateModelData testData;
	mat trainingData;
	unsigned int N = 100;
	unsigned int dim = 10;

public:
	void readRandomTrainingData(void);
	void readRandomTrainingDataWithGradients(void);


};


void SurrogateModelDataTest::readRandomTrainingData(void){

	mat testDataMatrix(N,dim+1,fill::randu);
	trainingData = testDataMatrix;

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(trainingData,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testData.readData(fileNameDataInput);

}

void SurrogateModelDataTest::readRandomTrainingDataWithGradients(void){

	mat testDataMatrix(N,2*dim+1,fill::randu);
	trainingData = testDataMatrix;

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(trainingData,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testData.readData(fileNameDataInput);

}


TEST_F(SurrogateModelDataTest, testIfConstructorWorks) {

	ASSERT_EQ(testData.getNumberOfSamples(),0);
	ASSERT_EQ(testData.getDimension(),0);
	ASSERT_FALSE(testData.isDataRead());

}

TEST_F(SurrogateModelDataTest, testreadData) {

	readRandomTrainingData();
	mat rawData = testData.getRawData();

	bool result = isEqual(rawData,trainingData, 10E-6);

	ASSERT_EQ(result,true);
	ASSERT_TRUE(testData.isDataRead());


}


TEST_F(SurrogateModelDataTest, testassignDimension) {

	readRandomTrainingData();
	testData.assignDimensionFromData();

	unsigned int dim = testData.getDimension();
	ASSERT_EQ(dim,10);

}

TEST_F(SurrogateModelDataTest, testassignDimensionIfDataHasGradients) {

	testData.setGradientsOn();
	readRandomTrainingDataWithGradients();
	testData.assignDimensionFromData();

	unsigned int dim = testData.getDimension();
	ASSERT_EQ(dim,10);

}

TEST_F(SurrogateModelDataTest, testassignSampleInputMatrix) {

	testData.setGradientsOn();
	readRandomTrainingDataWithGradients();
	testData.assignDimensionFromData();

	testData.assignSampleInputMatrix();
	rowvec x = testData.getRowX(7);

	for(unsigned int i=0; i<testData.getDimension(); i++){

		double error = fabs(x(i)- trainingData(7,i));
		EXPECT_LT(error, 10E-10 );
	}

}

TEST_F(SurrogateModelDataTest, testassignSampleOutputVector) {

	testData.setGradientsOn();
	readRandomTrainingDataWithGradients();
	testData.assignDimensionFromData();
	testData.assignSampleOutputVector();

	vec y = testData.getOutputVector();

	for(unsigned int i=0; i<N; i++){

		double error = fabs(y(i)- trainingData(i,dim));
		EXPECT_LT(error, 10E-10 );
	}

}

TEST_F(SurrogateModelDataTest, testnormalizeSampleInputMatrix) {

	testData.setGradientsOn();
	readRandomTrainingDataWithGradients();
	testData.assignDimensionFromData();
	testData.assignSampleOutputVector();

	vec lb(dim, fill::zeros);
	vec ub(dim, fill::ones);
	ub.fill(10.0);
	Bounds boxConstraints(lb,ub);
	testData.setBoxConstraints(boxConstraints);

	testData.normalizeSampleInputMatrix();

	ASSERT_TRUE(testData.isDataNormalized());

	unsigned int someRowIndex = 7;
	rowvec x = testData.getRowX(someRowIndex);

	for(unsigned int i=0; i<testData.getDimension(); i++){
		double normalizedValue = trainingData(someRowIndex,i)/100.0;
		double error = fabs(normalizedValue- x(i));
		EXPECT_LT(error, 10E-10 );
	}


}

TEST_F(SurrogateModelDataTest, testassignGradientMatrix) {


	testData.setGradientsOn();
	readRandomTrainingDataWithGradients();
	testData.assignDimensionFromData();
	testData.assignSampleOutputVector();
	testData.assignGradientMatrix();

	mat gradient = testData.getGradientMatrix();

	bool result = isEqual(gradient, trainingData.submat(0,dim+1,N-1, 2*dim), 10E-8);
	ASSERT_TRUE(result);

}







#endif
