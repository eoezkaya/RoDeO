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
#include "objective_function.hpp"
#include "optimization.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"

#define TEST_OBJECTIVE_FUNCTION
#ifdef TEST_OBJECTIVE_FUNCTION

TEST(testObjectiveFunction, testinitializeSurrogate){

	mat samples(100,5,fill::randu);
	saveMatToCVSFile(samples,"testObjectiveFunction.csv");
	vec lb(4); lb.fill(0.0);
	vec ub(4); ub.fill(1.0);

	ObjectiveFunction objFunTest("testObjectiveFunction",4);
	objFunTest.setParameterBounds(lb,ub);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	testObjectiveFunctionDef.outputFilename = "testObjectiveFunction.csv";

	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);

	objFunTest.initializeSurrogate();


	KrigingModel testModel = objFunTest.getSurrogateModel();

	mat rawData = testModel.getRawData();


	bool ifrawDataIsConsistent = isEqual(samples, rawData, 10E-10);
	ASSERT_TRUE(ifrawDataIsConsistent);
	samples(0,0) += 1000;
	ifrawDataIsConsistent = isEqual(samples, rawData, 10E-10);
	ASSERT_FALSE(ifrawDataIsConsistent);

	/* check dimension */
	ASSERT_EQ(testModel.getDimension(), 4);
	ASSERT_FALSE(testModel.areGradientsOn());
	ASSERT_TRUE(testModel.ifDataIsRead);


	remove("ObjectiveFunctionTest.csv");



}

TEST(testObjectiveFunction, testinitializeSurrogateWithAdjoint){

	unsigned int N = 100;
	unsigned dim = 2;

	mat samples(N,2*dim+1,fill::randu);
	saveMatToCVSFile(samples,"testObjectiveFunction.csv");
	vec lowerBounds(dim); lowerBounds.fill(0.0);
	vec upperBounds(dim); upperBounds.fill(1.0);

	ObjectiveFunction objFunTest("testObjectiveFunction",2);
	objFunTest.setParameterBounds(lowerBounds,upperBounds);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	testObjectiveFunctionDef.outputFilename = "testObjectiveFunction.csv";
	testObjectiveFunctionDef.ifGradient = true;
	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);


	objFunTest.initializeSurrogate();
	AggregationModel testModel = objFunTest.getSurrogateModelGradient();
	mat rawData = testModel.getRawData();

	bool ifrawDataIsConsistent = isEqual(samples, rawData, 10E-10);
	ASSERT_TRUE(ifrawDataIsConsistent);
	samples(0,0) += 1000;
	ifrawDataIsConsistent = isEqual(samples, rawData, 10E-10);
	ASSERT_FALSE(ifrawDataIsConsistent);


	ASSERT_EQ(testModel.getDimension(), dim);
	ASSERT_TRUE(testModel.areGradientsOn());
	ASSERT_TRUE(testModel.ifDataIsRead);


	remove("ObjectiveFunctionTest.csv");
}


TEST(testObjectiveFunction, testinitializeSurrogateWithML){

	unsigned int nSamplesLowFi = 200;
	unsigned int nSamplesHiFi  = 50;
	unsigned int dim = 2;

	generateHimmelblauDataMultiFidelity("highFidelityTestData.csv","lowFidelityTestData.csv",nSamplesHiFi,nSamplesLowFi);


	vec lowerBounds(dim); lowerBounds.fill(-6.0);
	vec upperBounds(dim); upperBounds.fill(6.0);


	ObjectiveFunction objFunTest("testObjectiveFunctionMLSurrogate",2);

//	objFunTest.setDisplayOn();
	objFunTest.setParameterBounds(lowerBounds,upperBounds);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("testObjectiveFunctionMLSurrogate");
	testObjectiveFunctionDef.outputFilename      = "highFidelityTestData.csv";
	testObjectiveFunctionDef.outputFilenameLowFi = "lowFidelityTestData.csv";
	testObjectiveFunctionDef.ifMultiLevel = true;


	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);

	objFunTest.initializeSurrogate();

	MultiLevelModel getTestModel = objFunTest.getSurrogateModelML();

	ASSERT_EQ(getTestModel.getDimension(), dim);


}


TEST(testObjectiveFunction, isMarkerFound){

	unsigned dim = 2;
	std::string marker = "Objective_function";
	ObjectiveFunction objFunTest("testObjectiveFunction",2);

	size_t found = objFunTest.isMarkerFound(marker,"Objective_function = 3.1");

	ASSERT_TRUE(found != std::string::npos);

	found = objFunTest.isMarkerFound(marker,"some irrelevant string");

	ASSERT_TRUE(found == std::string::npos);

	found = objFunTest.isMarkerFound(marker,"SomeObjective_function = 3.1");

	ASSERT_TRUE(found == std::string::npos);



}


TEST(testObjectiveFunction, getMarkerValue){

	unsigned dim = 2;
	ObjectiveFunction objFunTest("testObjectiveFunction",2);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	testObjectiveFunctionDef.marker = "Objective_function";
	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);
	double value = objFunTest.getMarkerValue("Objective_function = 3.1", 0);
	EXPECT_EQ(value,3.1);

}

TEST(testObjectiveFunction, getMarkerAdjointValues){

	unsigned dim = 2;
	ObjectiveFunction objFunTest("testObjectiveFunction",2);
	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	testObjectiveFunctionDef.markerForGradient = "gradient";
	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);
	rowvec gradient = objFunTest.getMarkerAdjointValues("gradient = 3.1, 2.7", 0);
	EXPECT_EQ(gradient(0),3.1);
	EXPECT_EQ(gradient(1),2.7);

}



TEST(testObjectiveFunction, readEvaluateOutputWithoutMarker){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.readEvaluateOutput(d);
	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	remove("readOutputTestFile.txt");

}

TEST(testObjectiveFunction, readEvaluateOutputWithoutMarkerWithAdjoint){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 1 2 3 4\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setGradientOn();
	objFunTest.readEvaluateOutput(d);
	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	ASSERT_EQ(d.gradient(0),1);
	ASSERT_EQ(d.gradient(1),2);
	ASSERT_EQ(d.gradient(2),3);
	ASSERT_EQ(d.gradient(3),4);
	remove("readOutputTestFile.txt");

}


TEST(testaObjectiveFunction, readEvaluateOutputWithOneMarker){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "myObjective = 2.144\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setReadMarker("myObjective");
	objFunTest.readEvaluateOutput(d);

	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	remove("readOutputTestFile.txt");

}


TEST(testaObjectiveFunction, readEvaluateOutputWithOneMarkerWithAdjoint){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "myObjective = 2.144\ngradient = 1,2,3,4\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setReadMarker("myObjective");
	objFunTest.setReadMarkerAdjoint("gradient");
	objFunTest.setGradientOn();
	objFunTest.readEvaluateOutput(d);

	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	ASSERT_EQ(d.gradient(0),1);
	ASSERT_EQ(d.gradient(1),2);
	ASSERT_EQ(d.gradient(2),3);
	ASSERT_EQ(d.gradient(3),4);
	remove("readOutputTestFile.txt");

}

TEST(testaObjectiveFunction, readEvaluateOutputWithTwoMarkers){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "myObjective = 2.144\ngradient = 1.0, 2.0, 3.0,   4.0";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setReadMarker("myObjective");
	objFunTest.setReadMarkerAdjoint("gradient");
	objFunTest.setGradientOn();
	objFunTest.readEvaluateOutput(d);

	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	ASSERT_EQ(d.gradient(0),1);
	ASSERT_EQ(d.gradient(1),2);
	ASSERT_EQ(d.gradient(2),3);
	ASSERT_EQ(d.gradient(3),4);
	remove("readOutputTestFile.txt");

}



TEST(testObjectiveFunction, testreadEvaluateOutputAdjoint){

	ObjectiveFunction objFunTest("testObjFun",4);

	Design d(4);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 2.944 -1.2 18.1 9.2\n";
	readOutputTestFile.close();

	objFunTest.setGradientOn();
	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.readEvaluateOutput(d);

	ASSERT_EQ(d.trueValue,2.144);
	ASSERT_EQ(d.objectiveFunctionValue,2.144);
	ASSERT_EQ(d.gradient(0),2.944);
	ASSERT_EQ(d.gradient(1),-1.2);
	ASSERT_EQ(d.gradient(2),18.1);
	ASSERT_EQ(d.gradient(3),9.2);
	remove("readOutputTestFile.txt");

}

TEST(testObjectiveFunction, calculateExpectedImprovement){


	mat samples(100,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(0.0,1.0);
		x(1) = generateRandomDouble(0.0,1.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}


	vec lb(2); lb.fill(0.0);
	vec ub(2); ub.fill(1.0);
	saveMatToCVSFile(samples,"EITest.csv");

	ObjectiveFunction objFunTest("EITest",2);
	objFunTest.setParameterBounds(lb,ub);


	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	testObjectiveFunctionDef.outputFilename = "EITest.csv";
	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);

	objFunTest.bindSurrogateModel();
	objFunTest.initializeSurrogate();

	CDesignExpectedImprovement testDesign(2,1);
	testDesign.generateRandomDesignVector();

	objFunTest.calculateExpectedImprovement(testDesign);
	EXPECT_GE(testDesign.valueExpectedImprovement, 0.0);

}

#endif


