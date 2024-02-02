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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */




#include "../INCLUDE/surrogate_model.hpp"
#include "../../Bounds/INCLUDE/bounds.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../../TestFunctions/INCLUDE/standard_test_functions.hpp"
#include<gtest/gtest.h>


class SurrogateModelBaseClassTestModel : public SurrogateModel{

private:


public:

	/* All pure virtual functions */

	void setBoxConstraints(Bounds boxConstraintsInput){

		assert(boxConstraintsInput.areBoundsSet());

		boxConstraints = boxConstraintsInput;
		data.setBoxConstraints(boxConstraintsInput);
	}

	void readData(void){

		data.readData(filenameDataInput);
		ifDataIsRead = true;

	}
	void normalizeData(void){


	}


	void setNameOfInputFile(string filename){

		filenameDataInput = filename;

	}
	void setNameOfHyperParametersFile(string filename){

		assert(isNotEmpty(filename));
	}
	void setNumberOfTrainingIterations(unsigned int){

	}

	void initializeSurrogateModel(void){

	}
	void printSurrogateModel(void) const {

	}
	void printHyperParameters(void) const {

	}
	void saveHyperParameters(void) const {

	}
	void loadHyperParameters(void) {

	}
	void updateAuxilliaryFields(void){

	}
	void train(void) {

	}
	double interpolate(rowvec x) const {

		return norm(x,2);
	}

	double interpolateUsingDerivatives(rowvec x) const {

		return norm(x,2);
	}

	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const {

		assert(!xp.is_empty());
		*f_tilde  = 0.0;
		*ssqr = 0.0;

	}

	void addNewSampleToData(rowvec newsample){

		assert(!newsample.is_empty());

	}
	void addNewLowFidelitySampleToData(rowvec newsample){

		assert(!newsample.is_empty());
	}

	void updateModelWithNewData(void){


	}



};

class SurrogateModelBaseTest : public ::testing::Test {
protected:
	void SetUp() override {

		himmelblauFunction.function.filenameTrainingData = "trainingData.csv";
		himmelblauFunction.function.numberOfTrainingSamples = 50;

		himmelblauFunction.function.filenameTestData = "testData.csv";
		himmelblauFunction.function.numberOfTestSamples = 10;



		testModel.setDimension(2);
		testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
		testModel.setNameOfInputFileTest(himmelblauFunction.function.filenameTestData);

		Bounds boxConstraints = himmelblauFunction.function.boxConstraints;
		testModel.setBoxConstraints(boxConstraints);

	}
	void TearDown() override {}


	SurrogateModelBaseClassTestModel testModel;
	HimmelblauFunction himmelblauFunction;

};

TEST_F(SurrogateModelBaseTest, ifConstructorWorks) {

	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModel.ifModelTrainingIsDone);

}



TEST_F(SurrogateModelBaseTest, readTestData) {

	himmelblauFunction.function.generateTrainingSamples();
	testModel.readData();

	himmelblauFunction.function.generateTestSamples();


	testModel.readDataTest();

	EXPECT_TRUE(testModel.ifHasTestData);
	EXPECT_TRUE(testModel.ifTestDataIsRead);

	remove("trainingData.csv");
	remove("testData.csv");
}



TEST_F(SurrogateModelBaseTest, tryOnTestData) {


	himmelblauFunction.function.generateTrainingSamples();
	testModel.readData();

	himmelblauFunction.function.generateTestSamples();
	testModel.setNameOfInputFileTest(himmelblauFunction.function.filenameTestData);

	testModel.readDataTest();
	testModel.normalizeDataTest();

	//	testModel.setDisplayOn();
	testModel.setNameOfOutputFileTest("surrogateTestResultsfile.csv");
	testModel.tryOnTestData();
	testModel.saveTestResults();

	mat results;
	results.load("surrogateTestResultsfile.csv", csv_ascii);

	ASSERT_EQ(results.n_rows, 11);
	ASSERT_EQ(results.n_cols, 5);


	remove("trainingData.csv");
	remove("testData.csv");
	remove("surrogateTestResultsfile.csv");

}

TEST_F(SurrogateModelBaseTest, reduceTrainingData) {


	himmelblauFunction.function.generateTrainingSamples();

	mat readDataInitial;
	readDataInitial.load("trainingData.csv", csv_ascii);
	//	readDataInitial.print("training data");

	testModel.readData();

	vec lb(2);
	vec ub(2);

	lb.fill(-3.0);
	ub.fill(3.0);

	testModel.reduceTrainingData(lb, ub);


	mat readDataReduced;
	readDataReduced.load("trainingData.csv", csv_ascii);

	//	readDataReduced.print("reduced training data");


	for(unsigned int i=0; i<readDataReduced.n_rows; i++){

		rowvec sample = readDataReduced.row(i);
		double x1 = sample(0);
		double x2 = sample(1);

		bool ifIsBetween = isNumberBetween(x1, lb(0),ub(0));
		ASSERT_TRUE(ifIsBetween);
		ifIsBetween = isNumberBetween(x2, lb(1),ub(1));
		ASSERT_TRUE(ifIsBetween);


	}

}

TEST_F(SurrogateModelBaseTest, countHowManySamplesAreWithinBounds) {


	himmelblauFunction.function.generateTrainingSamples();

	testModel.readData();

	vec lb(2);
	vec ub(2);

	lb.fill(-0.0000000001);
	ub.fill( 0.0000000001);

	unsigned int howMany =testModel.countHowManySamplesAreWithinBounds(lb, ub);

	ASSERT_TRUE(howMany == 0);

	lb.fill(-6.01);
	ub.fill( 6.01);

	howMany = testModel.countHowManySamplesAreWithinBounds(lb, ub);

	ASSERT_TRUE(howMany == himmelblauFunction.function.numberOfTrainingSamples);

	lb.fill(-3.0);
	ub.fill( 3.0);

	howMany = testModel.countHowManySamplesAreWithinBounds(lb, ub);

	ASSERT_TRUE(howMany < himmelblauFunction.function.numberOfTrainingSamples);

}
