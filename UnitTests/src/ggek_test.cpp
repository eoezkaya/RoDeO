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

#include "ggek.hpp"
#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "standard_test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef TEST_GGEK_MODEL

class GGEKModelTest: public ::testing::Test {
protected:
	void SetUp() override {
		vec ub(2);
		vec lb(2);
		ub.fill(6.0);
		lb.fill(-6.0);
		boxConstraints.setBounds(lb,ub);

		testModel.setBoxConstraints(boxConstraints);
		testModel.setDimension(2);
		testModel.setName("himmelblauModel");


	}

	void TearDown() override {}


	void generate2DHimmelblauDataForGGEKModel(unsigned int N) {


		himmelblauFunction.function.filenameTrainingData ="trainingSamplesHimmelblauGGEK.csv";
		himmelblauFunction.function.numberOfTrainingSamples = N;
		himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

		himmelblauFunction.function.filenameTestData = "testSamplesHimmelblauTGEK.csv";
		himmelblauFunction.function.numberOfTestSamples  = N;
		himmelblauFunction.function.generateTestSamples();


	}

	void generate2DHimmelblauDataForGGEKModelWithSomeZeroGradients(unsigned int N){

		himmelblauFunction.function.filenameTrainingData ="trainingSamplesHimmelblauGGEK.csv";
		himmelblauFunction.function.numberOfTrainingSamples = N;
		himmelblauFunction.function.generateTrainingSamplesWithAdjoints();

		mat data = himmelblauFunction.function.trainingSamples;
		for(unsigned int i=0; i<N; i++){

			if(i%5 == 0){

				data(i,3) = 0;
				data(i,4) = 0;
			}
		}
		data.save(himmelblauFunction.function.filenameTrainingData, csv_ascii);
	}

	GGEKModel testModel;
	HimmelblauFunction himmelblauFunction;
	Bounds boxConstraints;


};

TEST_F(GGEKModelTest, constructor) {

	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModel.ifNormalized);
	ASSERT_FALSE(testModel.ifHasTestData);


}

TEST_F(GGEKModelTest, readData) {

	unsigned int N = 50;
	unsigned int dim = 2;
	generate2DHimmelblauDataForGGEKModel(N);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();

	mat dataRead = testModel.getRawData();

	ASSERT_TRUE(testModel.getDimension() == dim);
	ASSERT_TRUE(dataRead.n_cols == 2*dim+1);
	ASSERT_TRUE(dataRead.n_rows == N);
	ASSERT_TRUE(testModel.ifDataIsRead);
}


TEST_F(GGEKModelTest, normalizeData) {

	unsigned int N = 50;
	unsigned int dim = 2;
	generate2DHimmelblauDataForGGEKModel(N);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	mat X = testModel.getX();

	ASSERT_TRUE(X.n_rows == N);
	ASSERT_TRUE(X.n_cols == dim);
	ASSERT_TRUE(testModel.ifNormalized);

}

TEST_F(GGEKModelTest, initializeSurrogateModel) {

	unsigned int N = 50;
	unsigned int dim = 2;
	generate2DHimmelblauDataForGGEKModel(N);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

//	testModel.printSurrogateModel();

	ASSERT_TRUE(testModel.ifInitialized);


}

TEST_F(GGEKModelTest, generateSampleWeights) {


	unsigned int N = 52;
	unsigned int dim = 2;
	generate2DHimmelblauDataForGGEKModel(N);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	testModel.generateSampleWeights();

	vec w = testModel.getSampleWeightsVector();

	ASSERT_TRUE(max(w) == 1.0);
	ASSERT_TRUE(w.size() == N);



}

TEST_F(GGEKModelTest, calculateIndicesOfSamplesWithActiveDerivatives) {


	unsigned int N = 52;
	unsigned int dim = 2;
	generate2DHimmelblauDataForGGEKModelWithSomeZeroGradients(N);

	testModel.setNameOfInputFile(himmelblauFunction.function.filenameTrainingData);
	testModel.readData();
	testModel.normalizeData();

	testModel.initializeSurrogateModel();

	testModel.calculateIndicesOfSamplesWithActiveDerivatives();



}



#endif
