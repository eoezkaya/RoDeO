/*
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


TEST(testDriver, testifFeatureIsOn){

	RoDeODriver testDriver;
	std::string s1 = "WARM_START";
	std::string s2 = "yes";
	testDriver.assignKeywordValue(s1,s2);
	bool ifWarmStart = testDriver.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "Yes";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "YES";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "Yeah";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, false);


}


TEST(testDriver, testifFeatureIsOff){

	RoDeODriver testDriver;
	std::string s1 = "WARM_START";
	std::string s2 = "N";
	testDriver.assignKeywordValue(s1,s2);
	bool ifWarmStart = testDriver.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "Yes";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, false);

	s2 = "NO";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "no";
	testDriver.assignKeywordValue(s1,s2);
	ifWarmStart = testDriver.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);


}


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


	int dimension = testDriver.getConfigKeyIntValue("DIMENSION");

	EXPECT_EQ(dimension, 2);

	std::string type = testDriver.getConfigKeyStringValue("PROBLEM_TYPE");

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




TEST(testDriver, testConfigKeysetValue){

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	EXPECT_EQ(testKey.doubleValue, 5.2);


	ConfigKey testKey2("TESTKEY2","int");
	number = "14";
	testKey2.setValue(number);
	EXPECT_EQ(testKey2.intValue, 14);



}


TEST(testDriver, testsearchConfigKeywordInString){

	RoDeODriver testDriver;
	std::string str = "DIMENSION=2";

	int foundIndex = testDriver.searchConfigKeywordInString(str);
	EXPECT_EQ(foundIndex, 2);

	str = "blablaDIMENSION=2";
	foundIndex = testDriver.searchConfigKeywordInString(str);
	EXPECT_EQ(foundIndex, -1);

	str = "NOTEXISTINGKEYWORD";

	foundIndex = testDriver.searchConfigKeywordInString(str);
	EXPECT_EQ(foundIndex, -1);


}

TEST(testDriver, testremoveKeywordFromString){

	RoDeODriver testDriver;
	std::string key = "DIMENSION";
	std::string s = "DIMENSION = blabla , blabla";

	std::string  s1 = testDriver.removeKeywordFromString(s, key);

	int ifEqual = s1.compare("blabla,blabla");
	ASSERT_EQ(ifEqual,0);

	key = "NOTAKEYWORD";

	std::string  s2 = testDriver.removeKeywordFromString(s, key);
	ifEqual = s2.compare(s);
	ASSERT_EQ(ifEqual,0);


}

TEST(testDriver, testassignKeywordValueWithIndex){

	RoDeODriver testDriver;
	std::string type = "MINIMIZATION";
	testDriver.assignKeywordValueWithIndex(type,0);

	ConfigKey testKey = testDriver.getConfigKey(0);
	int ifEqual = testKey.stringValue.compare("MINIMIZATION");
	ASSERT_EQ(ifEqual,0);

	testDriver.assignKeywordValueWithIndex("5",2);
	testKey = testDriver.getConfigKey("DIMENSION");

	ASSERT_EQ(testKey.intValue,5);


}



TEST(testDriver, testsetConfigKey){

	RoDeODriver testDriver;
	std::string type = "MINIMIZATION";
	testDriver.assignKeywordValue("PROBLEM_TYPE", type);

	ConfigKey testKey = testDriver.getConfigKey("PROBLEM_TYPE");

	int ifEqual = testKey.stringValue.compare("MINIMIZATION");
	ASSERT_EQ(ifEqual,0);


}



TEST(testDriver, testreadConfigFileForConstraintFunction){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileForConstraintFunction.cfg");


	testDriver.readConfigFile();


	std::string name = testDriver.getConfigKeyStringValue("PROBLEM_NAME");

	ASSERT_EQ(name, "HIMMELBLAU");

	std::string type = testDriver.getConfigKeyStringValue("PROBLEM_TYPE");

	ASSERT_EQ(type, "MINIMIZATION");

	int dim = testDriver.getConfigKeyIntValue("DIMENSION");
	ASSERT_EQ(dim, 2);

	vec lb = testDriver.getConfigKeyDoubleVectorValue("UPPER_BOUNDS");
	ASSERT_EQ(lb(0), 6.0);
	ASSERT_EQ(lb(1), 6.0);


}

TEST(testDriver, testreadConfigFile){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFile.cfg");
	testDriver.readConfigFile();


	std::string name = testDriver.getConfigKeyStringValue("PROBLEM_NAME");

	ASSERT_EQ(name, "HIMMELBLAU");

	std::string type = testDriver.getConfigKeyStringValue("PROBLEM_TYPE");

	ASSERT_EQ(type, "MINIMIZATION");

	int dim = testDriver.getConfigKeyIntValue("DIMENSION");
	ASSERT_EQ(dim, 2);

	vec lb = testDriver.getConfigKeyDoubleVectorValue("UPPER_BOUNDS");
	ASSERT_EQ(lb(0), 6.0);
	ASSERT_EQ(lb(1), 6.0);

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


	int dimension = testDriver.getConfigKeyIntValue("DIMENSION");
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





TEST(testDriver, testrunSurrogateModelTest){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest.cfg");
	testDriver.readConfigFile();

	compileWithCpp("himmelblau.cpp","himmelblau");

	testDriver.runSurrogateModelTest();

	mat results;
	results.load("ObjectiveFunction_TestResults.csv", csv_ascii);


	vec SE = results.col(4);
	SE(0) = 0.0;

	double sum = mean(SE);
	ASSERT_LT(sum,1000);

}



TEST(testDriver, testrunSurrogateModelTest2){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest2.cfg");
	testDriver.readConfigFile();

	TestFunction testFunctionEggholder("Himmelblau",2);

	testFunctionEggholder.setFunctionPointer(Himmelblau);

	testFunctionEggholder.setBoxConstraints(-6.0,6.0);
	mat samples = testFunctionEggholder.generateRandomSamples(100);

	saveMatToCVSFile(samples,"trainingInput.csv");

	samples = testFunctionEggholder.generateRandomSamples(100);

	saveMatToCVSFile(samples,"testInput.csv");


	testDriver.runSurrogateModelTest();

	mat results;
	results.load("ObjectiveFunction_TestResults.csv", csv_ascii);


	vec SE = results.col(4);
	SE(0) = 0.0;

	double sum = mean(SE);
	ASSERT_LT(sum,500);



}

TEST(testDriver, testrunSurrogateModelTest3){
	RoDeODriver testDriver;
	testDriver.setConfigFilename("testConfigFileSurrogateTest3.cfg");
	testDriver.readConfigFile();

	TestFunction testFunctionEggholder("Himmelblau",2);

	testFunctionEggholder.setFunctionPointer(Himmelblau);

	testFunctionEggholder.setBoxConstraints(-6.0,6.0);
	mat samples = testFunctionEggholder.generateRandomSamples(100);


	saveMatToCVSFile(samples,"testInput.csv");


	testDriver.runSurrogateModelTest();

	mat results;
	results.load("ObjectiveFunction_TestResults.csv", csv_ascii);


	vec SE = results.col(4);
	SE(0) = 0.0;

	double sum = mean(SE);
	ASSERT_LT(sum,500);



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

TEST(testDriver, testrunOptimization1){

	/* In this test we perform unconstrained minimization of the Himmelblau function with only function values.
	 * There is no warm start so a DoE is performed initially
	 * */

	RoDeODriver testDriver;
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


