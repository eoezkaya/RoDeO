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

#include "test_functions.hpp"
#include<gtest/gtest.h>


TEST(testTestFunctions, testGenerateSamplesInput){

	TestFunction testFun("testFunction",2);
	testFun.setBoxConstraints(0.0, 2.0);
	testFun.setNumberOfTrainingSamples(100);
	testFun.setNumberOfTestSamples(10);
	testFun.generateSamplesInputTrainingData();
	testFun.generateSamplesInputTestData();

	mat samplesInput = testFun.getTrainingSamplesInput();


	ASSERT_EQ(samplesInput.n_rows, 100);
	ASSERT_EQ(samplesInput.n_cols, 2);


	samplesInput = testFun.getTestSamplesInput();

	ASSERT_EQ(samplesInput.n_rows, 10);
	ASSERT_EQ(samplesInput.n_cols, 2);

}

TEST(testTestFunctions, testevaluate){

	TestFunction testFun("testFunction",2);
	testFun.setFunctionPointer(LinearTF1);

	rowvec dv(2); dv(0) = 1.0; dv(1) = 2.0;
	Design testDesign(dv);

	testFun.evaluate(testDesign);
	double expectedResult = 2*dv(0)+3*dv(1)+1.5;

	EXPECT_EQ(testDesign.trueValue, expectedResult);

}

TEST(testTestFunctions, testevaluateAdjoint){

	TestFunction testFun("testFunction",2);
	testFun.setFunctionPointer(LinearTF1Adj);

	rowvec dv(2); dv(0) = 1.0; dv(1) = 2.0;
	Design testDesign(dv);

	testFun.evaluateAdjoint(testDesign);
	double expectedResult = 2*dv(0)+3*dv(1)+1.5;

	EXPECT_EQ(testDesign.trueValue, expectedResult);
	EXPECT_EQ(testDesign.gradient(0), 2.0);
	EXPECT_EQ(testDesign.gradient(1), 3.0);

}





