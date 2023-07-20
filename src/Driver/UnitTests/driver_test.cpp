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


#include "../INCLUDE/drivers.hpp"
#include "../../TestFunctions/INCLUDE/standard_test_functions.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include<gtest/gtest.h>


class DriverTest : public ::testing::Test {
protected:
	void SetUp() override {

		himmelblauFunction.function.filenameTrainingData = "himmelblau.csv";
		himmelblauFunction.function.filenameTrainingDataHighFidelity = "himmelblau.csv";
		himmelblauFunction.function.filenameTrainingDataLowFidelity =  "himmelblauLowFi.csv";
		himmelblauFunction.function.filenameTestData = "himmelblauTest.csv";
		himmelblauFunction.function.numberOfTrainingSamples = 50;
		himmelblauFunction.function.numberOfTrainingSamplesLowFi = 200;
		himmelblauFunction.function.numberOfTestSamples = 200;

		constraint1.function.filenameTrainingData = "constraint1.csv";
		constraint1.function.numberOfTrainingSamples = 40;

		constraint2.function.filenameTrainingData = "constraint2.csv";
		constraint2.function.numberOfTrainingSamples = 40;

		alpine02Function.function.numberOfTrainingSamples = 100;
		alpine02Function.function.numberOfTestSamples = 100;

	}



	void TearDown() override {	}

	HimmelblauFunction himmelblauFunction;
	Alpine02_5DFunction alpine02Function;
	HimmelblauConstraintFunction1 constraint1;
	HimmelblauConstraintFunction2 constraint2;
	WingweightFunction wingweightFunction;

};


TEST_F(DriverTest, constructor){

	RoDeODriver testObject;
	ASSERT_TRUE(testObject.availableSurrogateModels.size() > 0);
	ASSERT_TRUE(testObject.configKeys.countNumberOfElements() > 0);
	ASSERT_TRUE(testObject.configKeysObjectiveFunction.countNumberOfElements() > 0);
	ASSERT_TRUE(testObject.configKeysConstraintFunction.countNumberOfElements() > 0);


}

TEST_F(DriverTest, extractConfigDefinitionsFromString){

	RoDeODriver testObject;

	std::string testString;

	testString = "DIMENSION = 3\n PROBLEM_TYPE = OPTIMIZATION";

	testObject.extractConfigDefinitionsFromString(testString);

	int dim = testObject.configKeys.getConfigKeyIntValue("DIMENSION");

	std::string type = testObject.configKeys.getConfigKeyStringValue("PROBLEM_TYPE");

	ASSERT_EQ(dim,3);
	ASSERT_EQ(type, "OPTIMIZATION");

}

TEST_F(DriverTest, extractObjectiveFunctionDefinitionFromString){

	RoDeODriver testObject;

	std::string testString;

	testString = "this is a test\n";
	testString +="OBJECTIVE_FUNCTION{\n";
	testString +="NAME = HimmelblauFunction\n";
	testString +="DESIGN_VECTOR_FILE = dv.dat\n";
	testString +="OUTPUT_FILE = objFunVal.dat\n";
	testString +="}\ntest end";



	testObject.extractObjectiveFunctionDefinitionFromString(testString);


	std::string type = testObject.configKeysObjectiveFunction.getConfigKeyStringValue("NAME");

	ASSERT_EQ(type, "HimmelblauFunction");
}


TEST_F(DriverTest, runSurrogateModelHimmelblauOrdinaryKrigingAlpine02){
	alpine02Function.function.generateTrainingSamples();
	alpine02Function.function.generateTestSamples();

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/Alpine02_OrdinaryKriging.cfg");
	testDriver.readConfigFile();
	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTestResults.csv");

	ASSERT_TRUE(results.n_rows == alpine02Function.function.numberOfTestSamples+1);
	ASSERT_TRUE(results.n_cols == 8);



}


TEST_F(DriverTest, runSurrogateModelHimmelblauMLModel){



	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();
	himmelblauFunction.function.generateTestSamples();

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/Himmelblau_MF.cfg");
	testDriver.readConfigFile();
	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTestResults.csv");

	ASSERT_EQ(results.n_cols, 5);
	ASSERT_EQ(results.n_rows, 201);

}


TEST_F(DriverTest, runOptimizationHimmelblauUnconstrained){

	/* In this test we perform unconstrained minimization of the Himmelblau function
	 * with only function values */

	RoDeODriver testDriver;
	himmelblauFunction.function.generateTrainingSamples();


	compileWithCpp("../../../src/Driver/UnitTests/Auxiliary/himmelblau.cpp","himmelblau");

//	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFileHimmelblauUnconstrainedOptimization.cfg");
	testDriver.readConfigFile();
	testDriver.runOptimization();


}

TEST_F(DriverTest, runOptimizationHimmelblauConstrained){


	himmelblauFunction.function.generateTrainingSamples();
	constraint1.function.generateTrainingSamples();

	compileWithCpp("../../../src/Driver/UnitTests/Auxiliary/himmelblau.cpp","himmelblau");
	compileWithCpp("../../../src/Driver/UnitTests/Auxiliary/constraint1.cpp","constraint1");

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFileHimmelblauConstrainedOptimization1.cfg");
	testDriver.readConfigFile();
	testDriver.runOptimization();


}


TEST_F(DriverTest, runSurrogateModelHimmelblauGradientEnhanced){

	himmelblauFunction.function.ifSomeAdjointsAreLeftBlank = false;
	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	himmelblauFunction.function.generateTestSamples();

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFileSurrogateTestHimmelblauGradientEnhanced.cfg");
	testDriver.readConfigFile();
	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTestResults.csv");

	ASSERT_EQ(results.n_cols, 5);
	ASSERT_EQ(results.n_rows, 201);

}


TEST_F(DriverTest, runSurrogateModelHimmelblauTangentModel){


	himmelblauFunction.function.generateTrainingSamplesWithTangents();
	himmelblauFunction.function.generateTestSamples();

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFileSurrogateTestHimmelblauTangentEnhanced.cfg");
	testDriver.readConfigFile();
	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTestResults.csv");

	ASSERT_EQ(results.n_cols, 5);
	ASSERT_EQ(results.n_rows, 201);




}


TEST_F(DriverTest, runSurrogateModelHimmelblauOrdinaryKriging){


	himmelblauFunction.function.generateTrainingSamples();
	himmelblauFunction.function.generateTestSamples();

	RoDeODriver testDriver;

//	testDriver.setDisplayOn();
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFileSurrogateTestHimmelblauOrdinaryKriging.cfg");
	testDriver.readConfigFile();
	testDriver.runSurrogateModelTest();

}


TEST_F(DriverTest, parseObjectiveFunctionDefinition){

	std::string inputString = "NAME = ObjFun\n";
	inputString+="DESIGN_VECTOR_FILE = dv.dat\n";
	inputString+="EXECUTABLE = himmelblau\n";
	inputString+="OUTPUT_FILE = objFunVal.dat\n";
	inputString+="PATH = ./\n";
	inputString+="SURROGATE_MODEL = UNIVERSAL_KRIGING";
#if 0
	std::cout<<inputString<<"\n";
#endif
	RoDeODriver testDriver;
	testDriver.parseObjectiveFunctionDefinition(inputString);

	ObjectiveFunctionDefinition ObjDef = testDriver.getObjectiveFunctionDefinition();

	EXPECT_EQ(ObjDef.name, "ObjFun");
	EXPECT_EQ(ObjDef.designVectorFilename, "dv.dat");
	EXPECT_EQ(ObjDef.outputFilename, "objFunVal.dat");
	EXPECT_EQ(ObjDef.executableName, "himmelblau");
	EXPECT_EQ(ObjDef.path, "./");

}


TEST_F(DriverTest, extractConfigDefinitionFromString){

	std::string inputString = "DIMENSION = 2\nPROBLEM_TYPE = optimization\n";

#if 0
	std::cout<<"inputString =\n";
	std::cout<<inputString<<"\n";
#endif
	RoDeODriver testDriver;

	testDriver.extractConfigDefinitionsFromString(inputString);

#if 0
	testDriver.printObjectiveFunctionDefinition();
#endif


	unsigned int dimension = testDriver.getDimension();

	EXPECT_EQ(dimension, 2);

	std::string type = testDriver.getProblemType();

	EXPECT_EQ(type, "optimization");

}



TEST_F(DriverTest, extractConstraintDefinitionsFromString){

	std::string inputString = "CONSTRAINT_FUNCTION{\nDEFINITION = constraint1 > 1.0\nDESIGN_VECTOR_FILE = dv.dat\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nPATH = ./\n}\n#some comment\nCONSTRAINT_FUNCTION{\nDEFINITION = constraint2 < 0.0\nDESIGN_VECTOR_FILE = dv.dat\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nPATH = ./\n}\n#some comment again";

#if 0
	std::cout<<"inputString =\n";
	std::cout<<inputString<<"\n";
#endif
	RoDeODriver testDriver;

	testDriver.extractConstraintDefinitionsFromString(inputString);
#if 0
	testDriver.printAllConstraintDefinitions();
#endif



	ObjectiveFunctionDefinition constraint1 = testDriver.getConstraintDefinition(0);
	ObjectiveFunctionDefinition constraint2 = testDriver.getConstraintDefinition(1);



	EXPECT_EQ(constraint1.designVectorFilename, "dv.dat");
	EXPECT_EQ(constraint1.executableName, "himmelblau");


	EXPECT_EQ(constraint2.designVectorFilename, "dv.dat");
	EXPECT_EQ(constraint2.executableName, "himmelblau");


}



TEST_F(DriverTest, readConfigFile){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("../../../src/Driver/UnitTests/Auxiliary/testConfigFile.cfg");
	testDriver.readConfigFile();


	std::string name = testDriver.getProblemName();

	ASSERT_EQ(name, "HIMMELBLAU");

	std::string type = testDriver.getProblemType();

	ASSERT_EQ(type, "OPTIMIZATION");

	int dim = testDriver.getDimension();
	ASSERT_EQ(dim, 2);


	ObjectiveFunctionDefinition objFun= testDriver.getObjectiveFunctionDefinition();


	EXPECT_EQ(objFun.name, "HimmelblauFunction");
	EXPECT_EQ(objFun.designVectorFilename, "dv.dat");
	EXPECT_EQ(objFun.executableName, "himmelblau");
	EXPECT_EQ(objFun.outputFilename, "objFunVal.dat");
	EXPECT_EQ(objFun.path, "./");

}





