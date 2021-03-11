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
	std::string filenameHyperParams = testModel.getNameOfHyperParametersFile();
	std::string filenameDataInput = testModel.getNameOfInputFile();
	ASSERT_TRUE(filenameHyperParams == "testKrigingModel_kriging_hyperparameters.csv");
	ASSERT_TRUE(filenameDataInput == "testKrigingModel.csv");
	ASSERT_TRUE(testModel.ifUsesGradientData == false);


}

TEST(testKriging, testReadDataAndNormalize){

	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(10);
	saveMatToCVSFile(samples,"Eggholder.csv");
	KrigingModel testModel("Eggholder");

	testModel.readData();
	testModel.setParameterBounds(0.0, 200.0);
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




