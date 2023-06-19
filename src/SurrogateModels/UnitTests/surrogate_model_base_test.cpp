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

		return -21.1;
	}
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const {

	}

	void addNewSampleToData(rowvec newsample){

	}
	void addNewLowFidelitySampleToData(rowvec newsample){

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

	Bounds newBoxConstraints(2);
	newBoxConstraints.setBounds(lb,ub);

	testModel.setBoxConstraints(newBoxConstraints);

	/* reduce size of training data around the 5.0 */
	testModel.reduceTrainingData(10, 5.0);


	mat readDataReduced;
	readDataReduced.load("trainingData.csv", csv_ascii);

//	readDataReduced.print("reduced training data");

	ASSERT_TRUE(readDataReduced.n_rows + 10 ==  readDataInitial.n_rows);


}


