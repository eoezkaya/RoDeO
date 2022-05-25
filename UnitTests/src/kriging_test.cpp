/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>


mat generate2DEggholderTestDataForKrigingModel(unsigned int N){

	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(N);
	saveMatToCVSFile(samples,"Eggholder.csv");

	return samples;
}


class KrigingModelTest : public ::testing::Test {
protected:
	void SetUp() override {


		unsigned int N = 100;
		generate2DEggholderTestDataForKrigingModel(N);
		testModel.setName("Eggholder");
		testModel.setNameOfInputFile("Eggholder.csv");
		testModel.readData();
		testModel.setBoxConstraints(0.0, 200.0);
		testModel.normalizeData();
		testModel.initializeSurrogateModel();

		N = 10;
		mat testData1D(N,2);
		testData1D(0,0) = 0.0;
		testData1D(1,0) = 0.1;
		testData1D(2,0) = 0.15;
		testData1D(3,0) = 0.2;
		testData1D(4,0) = 0.25;
		testData1D(5,0) = 0.4;
		testData1D(6,0) = 0.5;
		testData1D(7,0) = 0.75;
		testData1D(8,0) = 0.85;
		testData1D(9,0) = 1.0;

		testData1D(0,1) = 3.027209981231713;
		testData1D(1,1) = -0.656576774305574;
		testData1D(2,1) = -0.978280648621704;
		testData1D(3,1) =  -0.639727105946563;
		testData1D(4,1) =  -0.210367746201974;
		testData1D(5,1) =   0.114776974543924;
		testData1D(6,1) =   0.909297426825682;
		testData1D(7,1) =  -5.993276716644615;
		testData1D(8,1) =  -0.798489161076149;
		testData1D(9,1) =  15.829731945974109;

		saveMatToCVSFile(testData1D, "testdata1D.csv");

		testModel1D.setName("testdata1D");
		testModel1D.setNameOfInputFile("testdata1D.csv");
		testModel1D.readData();
		testModel1D.setBoxConstraints(0.0, 1.0);
		testModel1D.normalizeData();
		testModel1D.initializeSurrogateModel();



	}

	//  void TearDown() override {}

	KrigingModel testModel;
	KrigingModel testModel1D;
	KrigingHyperParameterOptimizer testOptimizer;
};

TEST_F(KrigingModelTest, testIfConstructorWorks) {

	ASSERT_TRUE(testModel.ifDataIsRead);
	ASSERT_TRUE(testModel.ifInitialized);
	ASSERT_TRUE(testModel.ifNormalized);
	ASSERT_FALSE(testModel.ifHasTestData);
	ASSERT_FALSE(testModel.areGradientsOn());

	ASSERT_TRUE(testModel.getDimension() == 2);
	std::string filenameDataInput = testModel.getNameOfInputFile();
	ASSERT_TRUE(filenameDataInput == "Eggholder.csv");


}



TEST_F(KrigingModelTest, testcalculateLikelihood) {

	vec hyperParameters(2);
	hyperParameters(0) = 5.0;
	hyperParameters(1) = 2.0;
	double L = testModel1D.calculateLikelihoodFunction(hyperParameters);

	double error = fabs(L - -24.202852114659308);

	ASSERT_LT(error,0.01);



}




TEST_F(KrigingModelTest, testKrigingOptimizerinitializeKrigingModelObject) {

	KrigingHyperParameterOptimizer testOptimizer;

	testOptimizer.initializeKrigingModelObject(testModel);
	ASSERT_TRUE(testOptimizer.ifModelObjectIsSet);

}



TEST_F(KrigingModelTest, testKrigingOptimizertestKrigingOptimizerOptimize) {

	unsigned int dim = 2;

	vec hyperParameters(2*dim);
	hyperParameters(0) = 10.0;
	hyperParameters(dim-1) = 0.1;
	hyperParameters(dim) = 0.5;
	hyperParameters(dim+1) = 1.5;
	double L = testModel.calculateLikelihoodFunction(hyperParameters);

	KrigingHyperParameterOptimizer testOptimizer;
	testOptimizer.initializeKrigingModelObject(testModel);

	testOptimizer.setDimension(2*dim);
	Bounds boxConstraints(2*dim);
	vec lb(2*dim); lb(0) = 0.0; lb(1) = 0.0; lb(2) = 0.0; lb(3) = 0.0;
	vec ub(2*dim); ub(0) = 10.0; ub(1) = 10.0; ub(2) = 2.0; ub(3) = 2.0;
	boxConstraints.setBounds(lb,ub);
	testOptimizer.setBounds(boxConstraints);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(100*2*dim);
	testOptimizer.setNumberOfDeathsInAGeneration(100*dim);
	testOptimizer.setInitialPopulationSize(2*dim*100);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.setMaximumNumberOfGeneratedIndividuals(100000*2*dim);
	testOptimizer.setNumberOfGenerations(5);
//	testOptimizer.setDisplayOn();


	testOptimizer.optimize();

	vec optimizedHyperParameters = testOptimizer.getBestDesignvector();

	double Loptimized = testModel.calculateLikelihoodFunction(optimizedHyperParameters);

	ASSERT_TRUE(Loptimized>L);

}

TEST_F(KrigingModelTest, testKrigingTrain) {

//	testModel.setnumberOfThreadsUsedForTraining(4);
	testModel.train2();

	abort();
}











TEST(testKriging, testInSampleErrorCloseToZeroWithoutTraining){


	mat samples(10,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}


	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);
	saveMatToCVSFile(samples,"KrigingTest.csv");

	KrigingModel testModel("KrigingTest");
	testModel.readData();
	testModel.setBoxConstraints(lb, ub);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();



	rowvec xp(2); xp(0) = samples(0,0); xp(1) = samples(0,1);

	rowvec xpnorm = normalizeRowVector(xp,lb,ub);

	double ftilde = testModel.interpolate(xpnorm);

	double error = fabs(ftilde - samples(0,2));
	EXPECT_LT(error, 10E-6);


}

TEST(testKriging, testInSampleErrorCloseToZeroAfterTraining){


	mat samples(10,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}


	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);
	saveMatToCVSFile(samples,"KrigingTest.csv");

	KrigingModel testModel("KrigingTest");
	testModel.readData();
	testModel.setBoxConstraints(lb, ub);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(100);

	rowvec xp(2); xp(0) = samples(0,0); xp(1) = samples(0,1);

	rowvec xpnorm = normalizeRowVector(xp,lb,ub);

	double ftilde = testModel.interpolate(xpnorm);

	double error = fabs(ftilde - samples(0,2));
	EXPECT_LT(error, 10E-6);


}



