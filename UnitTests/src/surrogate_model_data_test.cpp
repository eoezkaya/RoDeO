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


TEST(testSurrogateModelData, testConstructor){

	SurrogateModelData dataTest;
	ASSERT_EQ(dataTest.getNumberOfSamples(),0);
	ASSERT_EQ(dataTest.getDimension(),0);

}

TEST(testSurrogateModelData, testreadData){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testData(N,dim+1,fill::randu);

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testData,fileNameDataInput);

	SurrogateModelData dataTest;

//	dataTest.setDisplayOn();
	dataTest.readData(fileNameDataInput);
	mat rawData = dataTest.getRawData();

	bool result = isEqual(rawData,testData, 10E-6);

	ASSERT_EQ(result,true);

}


TEST(testSurrogateModelData, testassignDimension){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testData(N,dim+1,fill::randu);

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testData,fileNameDataInput);

	SurrogateModelData dataTest;

	//	dataTest.setDisplayOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();

	unsigned int dimGet = dataTest.getDimension();
	ASSERT_EQ(dimGet,dim);

}

TEST(testSurrogateModelData, testassignDimensionWithGradient){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testDataWithGradient(N,2*dim+1,fill::randu);

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataWithGradient,fileNameDataInput);

	SurrogateModelData dataTest;

	//	dataTest.setDisplayOn();
	dataTest.setGradientsOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();

	unsigned int dimGet = dataTest.getDimension();
	ASSERT_EQ(dimGet,dim);


}

TEST(testSurrogateModelData, testassignSampleInputMatrix){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testDataWithGradient(N,2*dim+1,fill::randu);

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataWithGradient,fileNameDataInput);

	SurrogateModelData dataTest;

	//	dataTest.setDisplayOn();
	dataTest.setGradientsOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();


	dataTest.assignSampleInputMatrix();
	rowvec x = dataTest.getRowX(7);


	for(unsigned int i=0; i<dataTest.getDimension(); i++){

		double error = fabs(x(i)- testDataWithGradient(7,i));
		EXPECT_LT(error, 10E-10 );
	}


}

TEST(testSurrogateModelData, testsetBoxConstraints){

	unsigned int dim = 5;
	SurrogateModelData dataTest;

	vec lb(dim, fill::zeros);
	vec ub(dim, fill::ones);
	Bounds boxConstraints(lb,ub);
	dataTest.setBoxConstraints(boxConstraints);


}

TEST(testSurrogateModelData, testsetBoxConstraintsFromData){

	unsigned int dim = 5;
	unsigned int N = 100;

	mat testData(N,dim+1, fill::randu);

	testData = 10*testData;
	testData(5,2) = 14.0;

	saveMatToCVSFile(testData,"testData.csv");

	SurrogateModelData dataTest;
	dataTest.readData("testData.csv");

	dataTest.setBoxConstraintsFromData();

	Bounds boundsGet = dataTest.getBoxConstraints();

	double upperBound = boundsGet.getUpperBound(2);

	ASSERT_EQ(upperBound, 14.0);

	remove("testData.csv");

}



TEST(testSurrogateModelData, testisDataNormalized){

	SurrogateModelData dataTest;
	ASSERT_FALSE(dataTest.isDataNormalized());
}



TEST(testSurrogateModelData, testnormalizeSampleInputMatrix){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testDataWithGradient(N,2*dim+1,fill::randu);

	testDataWithGradient = 10.0*testDataWithGradient;

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataWithGradient,fileNameDataInput);

	SurrogateModelData dataTest;

	//	dataTest.setDisplayOn();
	dataTest.setGradientsOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();
	dataTest.assignSampleInputMatrix();

	vec lb(dim, fill::zeros);
	vec ub(dim, fill::ones);
	ub.fill(10.0);
	Bounds boxConstraints(lb,ub);
	dataTest.setBoxConstraints(boxConstraints);

	dataTest.normalizeSampleInputMatrix();

	ASSERT_TRUE(dataTest.isDataNormalized());

	unsigned int someRowIndex = 7;
	rowvec x = dataTest.getRowX(someRowIndex);

	for(unsigned int i=0; i<dataTest.getDimension(); i++){
		double normalizedValue = testDataWithGradient(someRowIndex,i)/100.0;
		double error = fabs(normalizedValue- x(i));
		EXPECT_LT(error, 10E-10 );
	}

	remove("testData.csv");

}

TEST(testSurrogateModelData, testassignSampleOutputVector){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testDataWithGradient(N,2*dim+1,fill::randu);

	testDataWithGradient = 10.0*testDataWithGradient;

	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataWithGradient,fileNameDataInput);

	SurrogateModelData dataTest;

	//	dataTest.setDisplayOn();
	dataTest.setGradientsOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();
	dataTest.assignSampleOutputVector();

	vec y = dataTest.getOutputVector();

	for(unsigned int i=0; i<N; i++){

		double error = fabs(y(i)- testDataWithGradient(i,dim));
		EXPECT_LT(error, 10E-10 );
	}

	remove("testData.csv");

}


TEST(testSurrogateModelData, testassignGradientMatrix){

	unsigned int N = 100;
	unsigned dim = 10;
	mat testDataWithGradient(N,2*dim+1,fill::randu);


	string fileNameDataInput = "testData.csv";
	saveMatToCVSFile(testDataWithGradient,fileNameDataInput);

	SurrogateModelData dataTest;

//	dataTest.setDisplayOn();
	dataTest.setGradientsOn();
	dataTest.readData(fileNameDataInput);
	dataTest.assignDimensionFromData();
	dataTest.assignGradientMatrix();

	mat gradient = dataTest.getGradientMatrix();

	bool result = isEqual(gradient, testDataWithGradient.submat(0,dim+1,N-1, 2*dim), 10E-8);
	ASSERT_TRUE(result);


	remove("testData.csv");
}





#endif
