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

#include "multi_level_method.hpp"
#include<gtest/gtest.h>
TEST(testMultiLevelModel, testConstructor){

	MultiLevelModel testModel("MLtestModel");
	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifBoundsAreSet);
	ASSERT_FALSE(testModel.ifInitialized);


}


TEST(testMultiLevelModel, testfindIndexHiFiToLowFiData){

	mat testHifiData(10,9,fill::randu);
	mat testLofiData(20,9,fill::randu);

	for(unsigned int i=0; i<10; i++){

		testLofiData.row(i+10) = testHifiData.row(i);
	}

	testLofiData = shuffle(testLofiData);

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	unsigned int index = testModel.findIndexHiFiToLowFiData(2);


	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();


	rowvec dx = dataLowFi.row(index) - dataHiFi.row(2);

	for(unsigned int i=0; i<9; i++){

		EXPECT_LT(dx(i),10E-10);
	}


}

TEST(testMultiLevelModel, prepareErrorData){
	mat testHifiData(100,9,fill::randu);
	mat testLofiData(200,9,fill::randu);

	testHifiData.save("HiFiData.csv", csv_ascii);
	testLofiData.save("LoFiData.csv", csv_ascii);

	MultiLevelModel testModel("MLtestModel");
	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
	testModel.setinputFileNameLowFidelityData("LoFiData.csv");

	testModel.setLowFidelityModel("ORDINARY_KRIGING");
	testModel.setErrorModel("ORDINARY_KRIGING");


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	testModel.prepareErrorData();

	ASSERT_TRUE(testModel.ifErrorDataIsSet);


}
//
//
//TEST(testMultiLevelModel, testReadData){
//	mat testHifiData(100,9,fill::randu);
//	mat testLofiData(200,9,fill::randu);
//
//	testHifiData.save("HiFiData.csv", csv_ascii);
//	testLofiData.save("LoFiData.csv", csv_ascii);
//
//	MultiLevelModel testModel("MLtestModel");
//	testModel.setinputFileNameHighFidelityData("HiFiData.csv");
//	testModel.setinputFileNameLowFidelityData("LoFiData.csv");
//
//	testModel.setLowFidelityModel("ORDINARY_KRIGING");
//	testModel.setErrorModel("ORDINARY_KRIGING");
//	testModel.readData();
//	ASSERT_TRUE(testModel.ifDataIsRead);
//
//
//
//}
//
//TEST(testMultiLevelModel, testSetParameterBounds){
//
//	mat testHifiData(100,9,fill::randu);
//	mat testLofiData(200,9,fill::randu);
//
//	testHifiData.save("HiFiData.csv", csv_ascii);
//	testLofiData.save("LoFiData.csv", csv_ascii);
//
//	MultiLevelModel testModel("MLtestModel");
//
//	vec lb(2); lb.fill(-2);
//	vec ub(2); ub.fill(-2);
//
//	testModel.setLowFidelityModel("ORDINARY_KRIGING");
//	testModel.setErrorModel("ORDINARY_KRIGING");
//
//	testModel.readData();
//	testModel.setParameterBounds(lb,ub);
//
//}
