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


#include "drivers.hpp"
#include "test_functions.hpp"
#include "matrix_vector_operations.hpp"
#include "auxiliary_functions.hpp"
#include<gtest/gtest.h>

#define TESTDRIVERON

#ifdef TESTDRIVERON


TEST(testDriver, testparseConstraintDefinition){

	std::string inputString = "DEFINITION = constraint1 > 1.0\nDESIGN_VECTOR_FILE = dv.dat\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nPATH = ./\nMARKER = someMarker\nMARKER_FOR_GRADIENT = someMarkerGradient\nGRADIENT=yes\n";

#if 0
	std::cout<<inputString<<"\n";
#endif
	RoDeODriver testDriver;
	testDriver.parseConstraintDefinition(inputString);
	ConstraintDefinition constraintDef = testDriver.getConstraintDefinition(0);

	EXPECT_EQ(constraintDef.value, 1.0);
	EXPECT_EQ(constraintDef.inequalityType, ">");
	EXPECT_EQ(constraintDef.name, "constraint1");
	EXPECT_EQ(constraintDef.designVectorFilename, "dv.dat");
	EXPECT_EQ(constraintDef.outputFilename, "objFunVal.dat");
	EXPECT_EQ(constraintDef.executableName, "himmelblau");
	EXPECT_EQ(constraintDef.path, "./");
	EXPECT_EQ(constraintDef.marker, "someMarker");
	EXPECT_EQ(constraintDef.markerForGradient, "someMarkerGradient");
	EXPECT_EQ(constraintDef.ifGradient, true);





}

TEST(testDriver, testparseObjectiveFunctionDefinition){

	std::string inputString = "NAME = ObjFun\nDESIGN_VECTOR_FILE = dv.dat\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nPATH = ./\nMARKER = someMarker\nMARKER_FOR_GRADIENT = someMarkerGradient\nGRADIENT= on\n";
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
	EXPECT_EQ(ObjDef.marker, "someMarker");
	EXPECT_EQ(ObjDef.markerForGradient, "someMarkerGradient");
	EXPECT_EQ(ObjDef.ifGradient, true);


}


TEST(testDriver, testextractConfigDefinitionFromString){

	std::string inputString = "DIMENSION = 2\nPROBLEM_TYPE = minimization\n";

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

	EXPECT_EQ(type, "minimization");

}



TEST(testDriver, testextractConstraintDefinitionsFromString){

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



	ConstraintDefinition constraint1 = testDriver.getConstraintDefinition(0);
	ConstraintDefinition constraint2 = testDriver.getConstraintDefinition(1);


	EXPECT_EQ(constraint1.name, "constraint1");
	EXPECT_EQ(constraint1.inequalityType, ">");
	EXPECT_EQ(constraint1.value, 1.0);
	EXPECT_EQ(constraint1.designVectorFilename, "dv.dat");
	EXPECT_EQ(constraint1.executableName, "himmelblau");

	EXPECT_EQ(constraint2.name, "constraint2");
	EXPECT_EQ(constraint2.inequalityType, "<");
	EXPECT_EQ(constraint2.value, 0.0);
	EXPECT_EQ(constraint2.designVectorFilename, "dv.dat");
	EXPECT_EQ(constraint2.executableName, "himmelblau");


}


TEST(testDriver, testextractObjectiveFunctionDefinitionFromString){

	std::string inputString = "OBJECTIVE_FUNCTION{\nNAME = objFun\nDESIGN_VECTOR_FILE = dv.dat\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nPATH = ./\n}";

#if 0
	std::cout<<"inputString =\n";
	std::cout<<inputString<<"\n";
#endif
	RoDeODriver testDriver;

	testDriver.extractObjectiveFunctionDefinitionFromString(inputString);

#if 0
	testDriver.printObjectiveFunctionDefinition();
#endif
	ObjectiveFunctionDefinition objFun= testDriver.getObjectiveFunctionDefinition();


	EXPECT_EQ(objFun.name, "objFun");
	EXPECT_EQ(objFun.designVectorFilename, "dv.dat");
	EXPECT_EQ(objFun.executableName, "himmelblau");
	EXPECT_EQ(objFun.outputFilename, "objFunVal.dat");
	EXPECT_EQ(objFun.path, "./");




}



TEST(testDriver, testreadConfigFileForConstraintFunction){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileForConstraintFunction.cfg");


	testDriver.readConfigFile();


	std::string name = testDriver.getProblemName();

	ASSERT_EQ(name, "HIMMELBLAU");

	std::string type = testDriver.getProblemType();

	ASSERT_EQ(type, "MINIMIZATION");

	int dim = testDriver.getDimension();
	ASSERT_EQ(dim, 2);


}

TEST(testDriver, testreadConfigFile){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFile.cfg");
	testDriver.readConfigFile();


	std::string name = testDriver.getProblemName();

	ASSERT_EQ(name, "HIMMELBLAU");

	std::string type = testDriver.getProblemType();

	ASSERT_EQ(type, "MINIMIZATION");

	int dim = testDriver.getDimension();
	ASSERT_EQ(dim, 2);


	ObjectiveFunctionDefinition objFun= testDriver.getObjectiveFunctionDefinition();


	EXPECT_EQ(objFun.name, "HimmelblauFunction");
	EXPECT_EQ(objFun.designVectorFilename, "dv.dat");
	EXPECT_EQ(objFun.executableName, "himmelblau");
	EXPECT_EQ(objFun.outputFilename, "objFunVal.dat");
	EXPECT_EQ(objFun.path, "./");
	EXPECT_EQ(objFun.ifGradient, false);




}

TEST(testDriver, testreadConfigFile2){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFile2.cfg");
	testDriver.readConfigFile();


	int dimension = testDriver.getDimension();
	EXPECT_EQ(dimension, 2);

	ObjectiveFunctionDefinition objFun= testDriver.getObjectiveFunctionDefinition();


	EXPECT_EQ(objFun.name, "HimmelblauFunction");
	EXPECT_EQ(objFun.designVectorFilename, "dv.dat");
	EXPECT_EQ(objFun.executableName, "himmelblau");
	EXPECT_EQ(objFun.outputFilename, "objFunVal.dat");
	EXPECT_EQ(objFun.path, "./");
	EXPECT_EQ(objFun.marker, "objFunVal");
	EXPECT_EQ(objFun.ifGradient, true);

}





TEST(testDriver, testrunSurrogateModelTestOrdinaryKriging){

	/* Here we test the ORDINARY_KRIGING model using the Himmelblau function */

	remove("trainingData.csv");
	remove("testDataInput.csv");

	unsigned int dim = 2;
	unsigned int N = 100;

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(Himmelblau);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);

	himmelblauFunction.setNumberOfTrainingSamples(N);
	himmelblauFunction.generateSamplesInputTrainingData();
	himmelblauFunction.generateTrainingSamples();
	mat trainingData = himmelblauFunction.getTrainingSamples();


	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();

	saveMatToCVSFile(trainingData,"trainingData.csv");
	saveMatToCVSFile(testDataInput,"testDataInput.csv");


	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest1.cfg");
	testDriver.readConfigFile();


	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	ASSERT_EQ(results.n_cols, dim+1);
	ASSERT_EQ(results.n_rows, N);



	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testDataInput.csv");

}



TEST(testDriver, testrunSurrogateModelTestAggregation){


	/* Here we test the AGGREGATION model using the Himmelblau function */

	remove("trainingData.csv");
	remove("testDataInput.csv");

	unsigned int dim = 2;
	unsigned int N = 100;

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(HimmelblauAdj);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);

	himmelblauFunction.setNumberOfTrainingSamples(N);
	himmelblauFunction.generateSamplesInputTrainingData();
	himmelblauFunction.generateTrainingSamples();
	mat trainingData = himmelblauFunction.getTrainingSamples();


	himmelblauFunction.setNumberOfTestSamples(N);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();

	saveMatToCVSFile(trainingData,"trainingData.csv");
	saveMatToCVSFile(testDataInput,"testDataInput.csv");


	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest2.cfg");
	testDriver.readConfigFile();


	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	ASSERT_EQ(results.n_cols, dim+1);
	ASSERT_EQ(results.n_rows, N);



	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("testDataInput.csv");



}

TEST(testDriver, testrunSurrogateModelTestMultiLevel){

	int nSamplesLowFi = 200;
	int nSamplesHiFi  = 50;

	unsigned int NTest = 100;
	unsigned int dim = 2;

	generateHimmelblauDataMultiFidelity("trainingData.csv","trainingDataLowFidelity.csv",nSamplesHiFi ,nSamplesLowFi);

	TestFunction himmelblauFunction("Himmelblau",dim);
	himmelblauFunction.setFunctionPointer(Himmelblau);
	himmelblauFunction.setBoxConstraints(-6.0, 6.0);



	himmelblauFunction.setNumberOfTestSamples(NTest);
	himmelblauFunction.generateSamplesInputTestData();
	himmelblauFunction.generateTestSamples();
	mat testData      = himmelblauFunction.getTestSamples();
	mat testDataInput = himmelblauFunction.getTestSamplesInput();

	saveMatToCVSFile(testDataInput,"testDataInput.csv");

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest3.cfg");
	testDriver.readConfigFile();


	testDriver.runSurrogateModelTest();

	mat results;
	results.load("surrogateTest.csv", csv_ascii);

	ASSERT_EQ(results.n_cols, dim+1);
	ASSERT_EQ(results.n_rows, NTest);


	remove("surrogateTest.csv");
	remove("trainingData.csv");
	remove("trainingDataLowFidelity.csv");
	remove("testDataInput.csv");


}


TEST(testDriver, testsetObjectiveFunction){

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest1.cfg");
	testDriver.readConfigFile();

	ObjectiveFunction testObjFun = testDriver.setObjectiveFunction();

	std::string command = testObjFun.getExecutionCommand();
	int dim = testObjFun.getDimension();

	ASSERT_EQ(command,"./himmelblauDoETest1");
	ASSERT_EQ(dim,2);


}


//TEST(testDriver, testsetObjectiveFunctionWithMultiLevel){
//
//	RoDeODriver testDriver;
//	testDriver.setConfigFilename("testConfigFilesetObjectiveFunctionML.cfg");
//	testDriver.readConfigFile();
//
//	ObjectiveFunction testObjFun = testDriver.setObjectiveFunction();
//
//	std::string command = testObjFun.getExecutionCommand();
//	int dim = testObjFun.getDimension();
//
//	ASSERT_EQ(command,"./himmelblauHF");
//	ASSERT_EQ(dim,2);
//
//	std::string commandLowFi = testObjFun.getExecutionCommandLowFi();
//	ASSERT_EQ(commandLowFi,"./himmelblauLF");
//
//
//}



TEST(testDriver, testcheckIfRunIsNecessary){

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileForCheckIfRunIsNecessary.cfg");
	//	testDriver.setDisplayOn();
	testDriver.readConfigFile();

#if 0
	testDriver.printAllConstraintDefinitions();
#endif


	bool ifRunIsNecessaryFor1 = testDriver.checkIfRunIsNecessary(0);

	ASSERT_EQ(ifRunIsNecessaryFor1,false);

	bool ifRunIsNecessaryFor2 = testDriver.checkIfRunIsNecessary(1);

	ASSERT_EQ(ifRunIsNecessaryFor2,false);

	bool ifRunIsNecessaryFor3 = testDriver.checkIfRunIsNecessary(2);

	ASSERT_EQ(ifRunIsNecessaryFor3,true);

	bool ifRunIsNecessaryFor4 = testDriver.checkIfRunIsNecessary(3);

	ASSERT_EQ(ifRunIsNecessaryFor4,false);



}




TEST(testDriver, testrunDoE1){

	/* In this test we perform the DoE of the Himmelblau function using only function values */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest1.cfg");

	testDriver.readConfigFile();
	//	testDriver.setDisplayOn();

	compileWithCpp("himmelblauDoETest1.cpp","himmelblauDoETest1");

	testDriver.runDoE();

	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = Himmelblau(x);
	double err = fabs(resultExpected- firstRowOfResults(2));

	EXPECT_LT(err,10E-08);



	remove("himmelblauDoETest1");



}

TEST(testDriver, testrunDoE2){

	/* In this test we perform the DoE of the Himmelblau function using only function values and gradient, DoE calls
	 *
	 * always the adjoint function.
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest2.cfg");
	testDriver.readConfigFile();

	//	testDriver.setDisplayOn();

	compileWithCpp("himmelblauDoETest2.cpp","himmelblauDoETest2");


	testDriver.runDoE();

	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	double xb[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = HimmelblauAdj(x,xb);
	double err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	double errSensitivity1 = fabs(xb[0]- firstRowOfResults(3));
	double errSensitivity2 = fabs(xb[1]- firstRowOfResults(4));
	EXPECT_LT(errSensitivity1,10E-08);
	EXPECT_LT(errSensitivity2,10E-08);


	remove("himmelblauDoETest2");


}

TEST(testDriver, testrunDoE3){

	/* In this test we perform the DoE of the Himmelblau function using only function values for objective function and
	 * two constraints (all three are in the same file)
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest3.cfg");
	testDriver.readConfigFile();
	//	testDriver.setDisplayOn();

	compileWithCpp("himmelblauDoETest3.cpp","himmelblauDoETest3");


	testDriver.runDoE();

	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = Himmelblau(x);
	double err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	mat resultsConstraint1;
	resultsConstraint1.load("Constraint1.csv",csv_ascii);
	//	resultsConstraint1.print();

	firstRowOfResults = resultsConstraint1.row(0);

	resultExpected = x[0]*x[0]+ x[1]*x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	mat resultsConstraint2;
	resultsConstraint2.load("Constraint2.csv",csv_ascii);
	//	resultsConstraint2.print();

	firstRowOfResults = resultsConstraint2.row(0);

	resultExpected = x[0]+ x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);


	remove("himmelblauDoETest3");


}


TEST(testDriver, testrunDoE4){

	/* In this test we perform the DoE of the Himmelblau function with function values and gradients for the objective function and
	 * two constraints (all three are in the same file, only functional values for the constraints)
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest4.cfg");
	testDriver.readConfigFile();



	compileWithCpp("himmelblauDoETest4.cpp","himmelblauDoETest4");

	remove("objective_function.csv");
	remove("Constraint1.csv");
	remove("Constraint2.csv");

	testDriver.runDoE();

	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	double xb[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = HimmelblauAdj(x,xb);
	double err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	double errSensitivity1 = fabs(xb[0]- firstRowOfResults(3));
	double errSensitivity2 = fabs(xb[1]- firstRowOfResults(4));
	EXPECT_LT(errSensitivity1,10E-08);
	EXPECT_LT(errSensitivity2,10E-08);

	mat resultsConstraint1;
	resultsConstraint1.load("Constraint1.csv",csv_ascii);

	firstRowOfResults = resultsConstraint1.row(0);

	resultExpected = x[0]*x[0]+ x[1]*x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	mat resultsConstraint2;
	resultsConstraint2.load("Constraint2.csv",csv_ascii);

	firstRowOfResults = resultsConstraint2.row(0);

	resultExpected = x[0]+ x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);


	remove("himmelblauDoETest4");

}

TEST(testDriver, testrunDoE5){

	/* In this test we perform the DoE of the Himmelblau function with function values and gradients for the objective function and
	 * two constraints (all three are in the same file, functional values and gradients for the constraint1, functional values
	 * for the constraint2)
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest5.cfg");
	testDriver.readConfigFile();


	compileWithCpp("himmelblauDoETest5.cpp","himmelblauDoETest5");

	remove("objective_function.csv");
	remove("Constraint1.csv");
	remove("Constraint2.csv");


	testDriver.runDoE();

	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	double xb[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = HimmelblauAdj(x,xb);
	double err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	double errSensitivity1 = fabs(xb[0]- firstRowOfResults(3));
	double errSensitivity2 = fabs(xb[1]- firstRowOfResults(4));
	EXPECT_LT(errSensitivity1,10E-08);
	EXPECT_LT(errSensitivity2,10E-08);

	mat resultsConstraint1;
	resultsConstraint1.load("Constraint1.csv",csv_ascii);


	firstRowOfResults = resultsConstraint1.row(0);

	resultExpected = x[0]*x[0]+ x[1]*x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	errSensitivity1 = fabs(2.0*x[0]- firstRowOfResults(3));
	errSensitivity2 = fabs(2.0*x[1]- firstRowOfResults(4));
	EXPECT_LT(errSensitivity1,10E-08);
	EXPECT_LT(errSensitivity2,10E-08);

	mat resultsConstraint2;
	resultsConstraint2.load("Constraint2.csv",csv_ascii);


	firstRowOfResults = resultsConstraint2.row(0);

	resultExpected = x[0]+ x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);


	remove("himmelblauDoETest5");


}

TEST(testDriver, testrunDoE6){

	/* In this test we perform the DoE of the Himmelblau function with only function values. The objective function and the
	 * first constraint are in the same file. The second constraint is computed with another executable and written to a
	 * different file
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETest6.cfg");
	testDriver.readConfigFile();


	compileWithCpp("himmelblauDoETest6.cpp","himmelblauDoETest6");
	compileWithCpp("himmelblauDoETestConstraint1.cpp","himmelblauDoETestConstraint1");

	remove("objective_function.csv");
	remove("Constraint1.csv");
	remove("Constraint2.csv");
	testDriver.runDoE();
	mat results;
	results.load("objective_function.csv",csv_ascii);

	rowvec firstRowOfResults = results.row(0);

	double x[2];
	double xb[2];
	x[0] = firstRowOfResults(0);
	x[1] = firstRowOfResults(1);
	double resultExpected = HimmelblauAdj(x,xb);
	double err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);

	mat resultsConstraint1;
	resultsConstraint1.load("Constraint1.csv",csv_ascii);


	firstRowOfResults = resultsConstraint1.row(0);

	resultExpected = x[0]*x[0]+ x[1]*x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);
	mat resultsConstraint2;
	resultsConstraint2.load("Constraint2.csv",csv_ascii);


	firstRowOfResults = resultsConstraint2.row(0);

	resultExpected = x[0]+ x[1];
	err = fabs(resultExpected- firstRowOfResults(2));
	EXPECT_LT(err,10E-08);
	remove("himmelblauDoETest6");
	remove("himmelblauDoETestConstraint1");


}




TEST(testDriver, testrunDoEHimmelblauOnlySamples){


	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoETestHimmelblauOnlySamples.cfg");
	testDriver.readConfigFile();


	testDriver.runDriver();

	mat results;
	results.load("himmelblau_samples.csv",csv_ascii);

	EXPECT_EQ(200,results.n_rows);
	EXPECT_EQ(2,results.n_cols);


}


TEST(testDriver, testrunDoEHighDimensionalFunctionOnlySamples){

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoEHighDimensionalFunctionOnlySamples.cfg");
	testDriver.setDisplayOn();
	testDriver.readConfigFile();


	testDriver.runDriver();

	mat results;
	results.load("highdimensional_function_samples.csv",csv_ascii);

	results.print();

	EXPECT_EQ(220,results.n_rows);
	EXPECT_EQ(7,results.n_cols);


}

TEST(testDriver, testrunDoEHighDimensionalFunctionOnlySamplesWithDiscreteValues){

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileDoEHighDimensionalFunctionOnlySamplesWithDiscreteValues.cfg");
	testDriver.setDisplayOn();
	testDriver.readConfigFile();


	testDriver.runDriver();

	mat results;
	results.load("highdimensional_function_samples_with_discrete_values.csv",csv_ascii);

	results.print();

	EXPECT_EQ(220,results.n_rows);
	EXPECT_EQ(7,results.n_cols);



}



TEST(testDriver, testrunOptimization1){

	/* In this test we perform unconstrained minimization of the Himmelblau function with only function values.
	 * There is no warm start so a DoE is performed initially
	 * */

	RoDeODriver testDriver;

	//	testDriver.setDisplayOn();

	testDriver.setConfigFilename("testConfigFileOptimizationTest1.cfg");
	testDriver.readConfigFile();


	compileWithCpp("himmelblau.cpp","himmelblau");
	testDriver.runOptimization();



}

TEST(testDriver, testrunOptimization2){

	/* In this test we perform unconstrained minimization of the Himmelblau function with only function values.
	 * There is warm start so a DoE is not performed initially
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileOptimizationTest2.cfg");
	testDriver.readConfigFile();
	compileWithCpp("himmelblau.cpp","himmelblau");
	testDriver.runOptimization();



}

TEST(testDriver, testrunOptimization3){

	/* In this test we perform minimization of the Himmelblau function with only function values.
	 * We have also a constraint. Its output is also in the same file
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileOptimizationTest3.cfg");
	testDriver.readConfigFile();
	compileWithCpp("himmelblauWithOneConstraint.cpp","himmelblauWithOneConstraint");
	testDriver.runOptimization();

}

TEST(testDriver, testrunOptimization4){

	/* In this test we perform minimization of the Himmelblau function with function values and adjoint for the objective function.
	 * We have here two constraints. The first constraint shares the same output file with the objective function
	 * */

	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileOptimizationTest4.cfg");
	testDriver.readConfigFile();
	compileWithCpp("himmelblauAdjWithOneConstraint.cpp","himmelblauAdjWithOneConstraint");
	compileWithCpp("Constraint2.cpp","Constraint2");
	testDriver.runOptimization();

}

#endif
