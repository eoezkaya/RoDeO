/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), Rheinland-Pfälzische Technische Universität
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


#include "../INCLUDE/globalOptimalDesign.hpp"
#include "../../Bounds/INCLUDE/bounds.hpp"


#include<gtest/gtest.h>



class GlobalDesignTest: public ::testing::Test {
protected:
	void SetUp() override {

		testDesign.setDimension(2);
		Bounds testBounds(2);
		testBounds.setBounds(-1,3);
		testDesign.setBoxConstraints(testBounds);

	}

	void TearDown() override {

	}


	GlobalOptimalDesign testDesign;




};

TEST_F(GlobalDesignTest, constructor){

	GlobalOptimalDesign testObject;

	ASSERT_TRUE(testObject.dimension == 0);
	ASSERT_TRUE(testObject.isDesignFeasible);
	ASSERT_EQ(testObject.ID,0);

}


TEST_F(GlobalDesignTest, setBoxConstraints){

	GlobalOptimalDesign testObject;
	testObject.setDimension(2);
	Bounds testBounds(2);
	testBounds.setBounds(-1,3);
	//	testBounds.print();
	testObject.setBoxConstraints(testBounds);

	ASSERT_TRUE(testObject.boxConstraints.areBoundsSet());

}


TEST_F(GlobalDesignTest, setGlobalOptimalDesignFromHistoryFile){


	mat historyFile;
	field<std::string> header;
	historyFile.load( csv_name("../../../src/Optimizers/UnitTests/Auxiliary/optimizationHistory.csv", header) );
	//	historyFile.print();
	testDesign.setGlobalOptimalDesignFromHistoryFile(historyFile);
	double err = fabs(testDesign.trueValue - 31.30778825);
	ASSERT_TRUE(err<10E-8);


}

TEST_F(GlobalDesignTest, setGradientGlobalOptimumFromTrainingData){
	rowvec dv(2);
	dv(0) = 3.18984765616132;
	dv(1) = 2.35926846167614;
	testDesign.designParameters = dv;
	std::string filename = "../../../src/Optimizers/UnitTests/Auxiliary/himmelblauTrainingDataWithGradients.csv";
	testDesign.setGradientGlobalOptimumFromTrainingData(filename);
	double err = fabs(testDesign.gradient(0) - 23.0899553754475);
	ASSERT_TRUE(err<10E-8);
	err = fabs(testDesign.gradient(1) - 18.1058541388457);
	ASSERT_TRUE(err<10E-8);
}

TEST_F(GlobalDesignTest, saveToXMLFile){
	rowvec dv(2);
	dv(0) = 3.18984765616132;
	dv(1) = 2.35926846167614;
	testDesign.designParameters = dv;

	testDesign.saveToXMLFile();


}


TEST_F(GlobalDesignTest, generateXmlString){
	rowvec dv(2);
	dv(0) = 3.18984765616132;
	dv(1) = 2.35926846167614;
	testDesign.designParameters = dv;

	std::string text = testDesign.generateXmlString();
	std::cout<<text<<"\n";

}


