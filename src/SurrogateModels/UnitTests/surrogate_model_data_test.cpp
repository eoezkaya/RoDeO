/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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

#include "../../Bounds/INCLUDE/bounds.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"

#include<gtest/gtest.h>
#include "../INCLUDE/surrogate_model_data.hpp"

#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
using std::string;


class SurrogateModelDataTest : public ::testing::Test {
protected:
	void SetUp() override {


	}

	void TearDown() override {




	}

	SurrogateModelData testSurrogateModelData;
	mat trainingData;
	mat testData;


public:
	void generateAndReadRandomTrainingData(unsigned int, unsigned int);
	void generateAndReadRandomTrainingDataWithGradients(unsigned int, unsigned int);
	void generateAndReadRandomTrainingDataWithDirectionalDerivatives(unsigned int, unsigned int);

	void generateAndReadRandomTestData(unsigned int, unsigned int);




};


void SurrogateModelDataTest::generateAndReadRandomTrainingData(unsigned int N=100, unsigned int dim=5){

	mat testDataMatrix(N,dim+1,fill::randu);
	trainingData = testDataMatrix;

	string fileNameDataInput = "trainingData.csv";
	saveMatToCVSFile(trainingData,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testSurrogateModelData.readData(fileNameDataInput);

	remove(fileNameDataInput.c_str());

}

void SurrogateModelDataTest::generateAndReadRandomTrainingDataWithGradients(unsigned int N=100, unsigned int dim=5){

	mat testDataMatrix(N,2*dim+1,fill::randu);
	trainingData = testDataMatrix;

	string fileNameDataInput = "trainingDataWithGradients.csv";;
	saveMatToCVSFile(trainingData,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testSurrogateModelData.setGradientsOn();
	testSurrogateModelData.readData(fileNameDataInput);
	remove(fileNameDataInput.c_str());
}

void SurrogateModelDataTest::generateAndReadRandomTrainingDataWithDirectionalDerivatives(unsigned int N=100, unsigned int dim=5){

	mat testDataMatrix(N,2*dim+2,fill::randu);
	testDataMatrix = testDataMatrix*15.0;
	trainingData = testDataMatrix;

	string fileNameDataInput = "trainingDataWithDirectionalDerivatives.csv";;
	saveMatToCVSFile(trainingData,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testSurrogateModelData.setDirectionalDerivativesOn();
	testSurrogateModelData.readData(fileNameDataInput);
	remove(fileNameDataInput.c_str());
}



void SurrogateModelDataTest::generateAndReadRandomTestData(unsigned int N = 100,unsigned int dim = 5){

	mat testDataMatrix(N,dim+1,fill::randu);
	testData = testDataMatrix;

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataMatrix,fileNameDataInput);

	//	dataTest.setDisplayOn();
	testSurrogateModelData.readDataTest(fileNameDataInput);
	remove(fileNameDataInput.c_str());
}



TEST_F(SurrogateModelDataTest, testIfConstructorWorks) {

	ASSERT_EQ(testSurrogateModelData.getNumberOfSamples(),0);
	ASSERT_EQ(testSurrogateModelData.getDimension(),0);
	ASSERT_FALSE(testSurrogateModelData.isDataRead());

}

TEST_F(SurrogateModelDataTest, testreadData) {

	generateAndReadRandomTrainingData();
	mat rawData = testSurrogateModelData.getRawData();

	bool result = isEqual(rawData,trainingData, 10E-6);

	ASSERT_EQ(result,true);
	ASSERT_TRUE(testSurrogateModelData.isDataRead());

}


TEST_F(SurrogateModelDataTest, testassignDimension) {

	generateAndReadRandomTrainingData();
	testSurrogateModelData.assignDimensionFromData();

	unsigned int dim = testSurrogateModelData.getDimension();
	ASSERT_EQ(dim,5);

}

TEST_F(SurrogateModelDataTest, testassignDimensionIfDataHasGradients) {

	testSurrogateModelData.setGradientsOn();
	generateAndReadRandomTrainingDataWithGradients();
	testSurrogateModelData.assignDimensionFromData();

	unsigned int dim = testSurrogateModelData.getDimension();
	ASSERT_EQ(dim,5);

}

TEST_F(SurrogateModelDataTest, testassignDimensionIfDataHasDirectionalDerivatives) {

	testSurrogateModelData.setDirectionalDerivativesOn();
	generateAndReadRandomTrainingDataWithDirectionalDerivatives();
	testSurrogateModelData.assignDimensionFromData();

	unsigned int dim = testSurrogateModelData.getDimension();
	ASSERT_EQ(dim,5);

}



TEST_F(SurrogateModelDataTest, testassignSampleInputMatrix) {

	testSurrogateModelData.setGradientsOn();
	generateAndReadRandomTrainingDataWithGradients();
	testSurrogateModelData.assignDimensionFromData();

	testSurrogateModelData.assignSampleInputMatrix();
	rowvec x = testSurrogateModelData.getRowX(7);

	for(unsigned int i=0; i<testSurrogateModelData.getDimension(); i++){

		double error = fabs(x(i)- trainingData(7,i));
		EXPECT_LT(error, 10E-10 );
	}

}

TEST_F(SurrogateModelDataTest, testassignSampleOutputVector) {

	unsigned int dim = 5;
	testSurrogateModelData.setGradientsOn();
	generateAndReadRandomTrainingDataWithGradients();
	testSurrogateModelData.assignDimensionFromData();
	testSurrogateModelData.assignSampleOutputVector();

	vec y = testSurrogateModelData.getOutputVector();

	for(unsigned int i=0; i<100; i++){

		double error = fabs(y(i)- trainingData(i,dim));
		EXPECT_LT(error, 10E-10 );
	}

}

TEST_F(SurrogateModelDataTest, testnormalizeSampleInputMatrix) {

	testSurrogateModelData.setGradientsOn();
	generateAndReadRandomTrainingDataWithGradients();
	testSurrogateModelData.assignDimensionFromData();
	testSurrogateModelData.assignSampleOutputVector();

	vec lb(5, fill::zeros);
	vec ub(5, fill::ones);
	ub.fill(10.0);
	Bounds boxConstraints(lb,ub);
	testSurrogateModelData.setBoxConstraints(boxConstraints);

	testSurrogateModelData.normalizeSampleInputMatrix();

	ASSERT_TRUE(testSurrogateModelData.isDataNormalized());

	unsigned int someRowIndex = 7;
	rowvec x = testSurrogateModelData.getRowX(someRowIndex);

	for(unsigned int i=0; i<testSurrogateModelData.getDimension(); i++){
		double normalizedValue = trainingData(someRowIndex,i)/50.0;
		double error = fabs(normalizedValue- x(i));
		EXPECT_LT(error, 10E-10 );
	}


}

TEST_F(SurrogateModelDataTest, testassignGradientMatrix) {

	unsigned int dim = 6;
	unsigned int N = 22;
	testSurrogateModelData.setGradientsOn();
	generateAndReadRandomTrainingDataWithGradients(N,dim);
	testSurrogateModelData.assignDimensionFromData();
	testSurrogateModelData.assignSampleOutputVector();
	testSurrogateModelData.assignGradientMatrix();

	mat gradient = testSurrogateModelData.getGradientMatrix();

	bool result = isEqual(gradient, trainingData.submat(0,dim+1,N-1, 2*dim), 10E-8);
	ASSERT_TRUE(result);

}

TEST_F(SurrogateModelDataTest, testreadDataTest) {

	unsigned int dim = 3;
	unsigned int N = 23;
	testSurrogateModelData.setDimension(dim);
	generateAndReadRandomTestData(N, dim);
	ASSERT_TRUE(testSurrogateModelData.ifTestDataIsRead);

	unsigned int numberOfTestSamples = testSurrogateModelData.getNumberOfSamplesTest();

	ASSERT_TRUE(numberOfTestSamples == N);

}

TEST_F(SurrogateModelDataTest, testreadDataTestWithDirectionalDerivatives) {

	unsigned int dim = 4;
	unsigned int N = 10;
	testSurrogateModelData.setDimension(dim);
	generateAndReadRandomTrainingDataWithDirectionalDerivatives(N, dim);

	ASSERT_TRUE(testSurrogateModelData.ifDataIsRead);

}

TEST_F(SurrogateModelDataTest, testnormalizeData) {

	unsigned int dim = 3;
	unsigned int N = 4;
	testSurrogateModelData.setDimension(dim);
	generateAndReadRandomTrainingDataWithDirectionalDerivatives(N, dim);
	ASSERT_TRUE(testSurrogateModelData.ifDataIsRead);

	Bounds boxConstraints(dim);
	boxConstraints.setBounds(0.0,15.0);
	testSurrogateModelData.setBoxConstraints(boxConstraints);
	testSurrogateModelData.normalize();


	ASSERT_TRUE(testSurrogateModelData.ifDataIsNormalized);
}

TEST_F(SurrogateModelDataTest, removeVeryCloseSamplesLineSearch) {

	unsigned int dim = 3;
	unsigned int N = 40;
	testSurrogateModelData.setDimension(dim);
	mat testDataMatrix(N,dim+1,fill::randu);

	testDataMatrix.save("trainingData.csv", csv_ascii);

	testSurrogateModelData.readData("trainingData.csv");

	ASSERT_TRUE(testSurrogateModelData.ifDataIsRead);

	Bounds boxConstraints(dim);
	boxConstraints.setBounds(0.0,1.0);
	testSurrogateModelData.setBoxConstraints(boxConstraints);
	testSurrogateModelData.normalize();
	ASSERT_TRUE(testSurrogateModelData.ifDataIsNormalized);

	mat trainingData = testSurrogateModelData.X;
	rowvec dv = trainingData.row(11);

	std::vector<rowvec> history;
	history.push_back(trainingData.row(11));
	history.push_back(trainingData.row(12));
	history.push_back(trainingData.row(13));
	history.push_back(trainingData.row(14));


	Design globalOptimal;
	globalOptimal.designParametersNormalized = dv;

	testSurrogateModelData.removeVeryCloseSamples(globalOptimal,history);


	trainingData = testSurrogateModelData.X;
	ASSERT_TRUE(trainingData.n_rows == 37);

}






TEST_F(SurrogateModelDataTest, removeVeryCloseSamples) {

	unsigned int dim = 3;
	unsigned int N = 40;
	testSurrogateModelData.setDimension(dim);
	mat testDataMatrix(N,dim+1,fill::randu);

	rowvec r = testDataMatrix.row(5);
	r = r + 0.00001;
	testDataMatrix.row(11) = r;
	r = r - 0.00002;
	testDataMatrix.row(18) = r;

	testDataMatrix.save("trainingData.csv", csv_ascii);

	testSurrogateModelData.readData("trainingData.csv");

	ASSERT_TRUE(testSurrogateModelData.ifDataIsRead);

	Bounds boxConstraints(dim);
	boxConstraints.setBounds(0.0,1.0);
	testSurrogateModelData.setBoxConstraints(boxConstraints);
	testSurrogateModelData.normalize();
	ASSERT_TRUE(testSurrogateModelData.ifDataIsNormalized);

	mat trainingData = testSurrogateModelData.X;
	rowvec dv = trainingData.row(11);

	Design globalOptimal;
	globalOptimal.designParametersNormalized = dv;
	testSurrogateModelData.removeVeryCloseSamples(globalOptimal);


	trainingData = testSurrogateModelData.X;
	ASSERT_TRUE(trainingData.n_rows == 38);

}





