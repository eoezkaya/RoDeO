/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


TEST(testKriging, testKrigingConstructor){

	KrigingModel testModel("testKrigingModel");
	std::string filenameDataInput = testModel.getNameOfInputFile();
	ASSERT_TRUE(filenameDataInput == "testKrigingModel.csv");
	ASSERT_FALSE(testModel.areGradientsOn());


}

TEST(testKriging, testReadDataAndNormalize){

	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(10);
	saveMatToCVSFile(samples,"Eggholder.csv");


	KrigingModel testModel("Eggholder");


	testModel.readData();
	testModel.setBoxConstraints(0.0, 200.0);


	testModel.normalizeData();


	unsigned int N = testModel.getNumberOfSamples();
	ASSERT_TRUE(N == 10);
	unsigned int d = testModel.getDimension();
	ASSERT_TRUE(d == 2);
	mat rawData = testModel.getRawData();
	bool ifBothMatricesAreEqual = isEqual(samples,rawData,10E-10);


	ASSERT_TRUE(ifBothMatricesAreEqual);

	remove("Eggholder.csv");
}



TEST(testKriging, testsetParameterBounds){


	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(10);
	saveMatToCVSFile(samples,"Eggholder.csv");


	KrigingModel testModel("Eggholder");


	testModel.readData();

	Bounds boxConstraints(2);
	boxConstraints.setBounds(0.0,2.0);

	testModel.setBoxConstraints(boxConstraints);


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



