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
#include "constraint_functions.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include "design.hpp"


#define TEST_CONSTRAIN_FUNCTION
#ifdef TEST_CONSTRAIN_FUNCTION
TEST(testDriver, testConstraintDefinition){

	std::string testDefinition = "constraint1 < 3.2";
	ConstraintDefinition testConstraint(testDefinition);
	ASSERT_EQ(testConstraint.value,3.2);
	ASSERT_EQ(testConstraint.inequalityType,"<");
	ASSERT_EQ(testConstraint.name,"constraint1");

	std::string testDefinition2 = "constraint2 > 0.2";
	ConstraintDefinition testConstraint2(testDefinition2);
	ASSERT_EQ(testConstraint2.value,0.2);
	ASSERT_EQ(testConstraint2.inequalityType,">");
	ASSERT_EQ(testConstraint2.name,"constraint2");


}


TEST(testConstraintFunctions, testConstructor){

	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	unsigned int dim = constraintFunTest.getDimension();
	std::string name = constraintFunTest.getName();

	ASSERT_EQ(dim,4);
	ASSERT_TRUE(name.compare("testConstraintFunction") == 0);


}

TEST(testConstraintFunctions, testConstructorWithFunctionPointer){

	ConstraintFunction constraintFunTest("testConstraintFunction",2);
	constraintFunTest.setFunctionPointer(Himmelblau);


	unsigned int dim = constraintFunTest.getDimension();
	std::string name = constraintFunTest.getName();

	ASSERT_EQ(dim,2);
	ASSERT_TRUE(name.compare("testConstraintFunction") == 0);
	ASSERT_TRUE(constraintFunTest.ifHasFunctionFunctionPointer());


}


TEST(testConstraintFunctions, checkFeasibility){

	ConstraintFunction constraintFunTest("testConstraintFunction",4);

	ConstraintDefinition def("testConstraintFunction > 18.2");

	constraintFunTest.setParametersByDefinition(def);
	bool ifFeasible = constraintFunTest.checkFeasibility(18.4);
	ASSERT_TRUE(ifFeasible);
	ifFeasible = constraintFunTest.checkFeasibility(18.1);
	ASSERT_FALSE(ifFeasible);

}


TEST(testConstraintFunctions, testEvaluateExternal){

	Design d(2);
	d.generateRandomDesignVector(0.0,1.0);
	d.saveDesignVector("dv.dat");

	std::string compileCommand = "g++ himmelblau.cpp -o himmelblau -lm";
	system(compileCommand.c_str());


	ConstraintFunction constraintFunTest("testConstraintFunction",2);
	constraintFunTest.setExecutableName("himmelblau");
	constraintFunTest.evaluate(d);

	std::ifstream testInput("objFunVal.dat");
	double functionValue = 0.0;
	double functionValueFromFunction = Himmelblau(d.designParameters.memptr());
	testInput >> functionValue;
	testInput.close();
	double error = fabs(functionValue - functionValueFromFunction);
	EXPECT_LT(error,10E-08);

	remove("himmelblau");
	remove("dv.dat");
	remove("objFunVal.dat");
}

TEST(testConstraintFunctions, testreadEvaluateOutputWithoutMarker){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144\n";
	readOutputTestFile.close();

	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(0);

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);
	remove("readOutputTestFile.txt");

}

TEST(testConstraintFunctions, testreadEvaluateOutputWithoutMarkerWithAdjoint){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 1 2 3 4\n";
	readOutputTestFile.close();

	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(0);
	constraintFunTest.setGradientOn();
	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);
	rowvec gradientResult = d.constraintGradients.front();
	EXPECT_EQ(gradientResult(0),1);
	EXPECT_EQ(gradientResult(1),2);
	EXPECT_EQ(gradientResult(2),3);
	EXPECT_EQ(gradientResult(3),4);

	remove("readOutputTestFile.txt");
}



TEST(testConstraintFunctions, testreadEvaluateOutputWithMarkerWithoutAdjoint){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "myConstraint = 2.144\n";
	readOutputTestFile.close();


	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(0);
	constraintFunTest.setReadMarker("myConstraint");

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);
	remove("readOutputTestFile.txt");



}

TEST(testConstraintFunctions, testreadEvaluateOutputWithMarkerWithAdjoint){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "myConstraint = 2.144\nConstraintGradient = 1,2,3,4\n";
	readOutputTestFile.close();

	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(0);
	constraintFunTest.setReadMarker("myConstraint");
	constraintFunTest.setReadMarkerAdjoint("ConstraintGradient");

	constraintFunTest.setGradientOn();

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);
	rowvec gradientResult = d.constraintGradients.front();
	EXPECT_EQ(gradientResult(0),1);
	EXPECT_EQ(gradientResult(1),2);
	EXPECT_EQ(gradientResult(2),3);
	EXPECT_EQ(gradientResult(3),4);


	remove("readOutputTestFile.txt");


}






TEST(testConstraintFunctions, testreadEvaluateThreeOutputsInTheSameFileWithMarkersAndTwoOfThemHasAdjoints){

	Design d(4);
	d.setNumberOfConstraints(3);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "Constraint1 = 2.144\nConstraint2 = -1\nConstraint2Gradient = 1,2,3,4\nConstraint3 = -1.6\nConstraint3Gradient = -1,-2,-3,-4";
	readOutputTestFile.close();

	/* first constraint has only value */
	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(0);
	constraintFunTest.setReadMarker("Constraint1");

	/* second constraint has value and gradient*/
	ConstraintFunction constraintFunTest2("testConstraintFunction2",4);
	constraintFunTest2.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest2.setID(1);
	constraintFunTest2.setGradientOn();
	constraintFunTest2.setReadMarker("Constraint2");
	constraintFunTest2.setReadMarkerAdjoint("Constraint2Gradient");

	/* third constraint has value and gradient*/
	ConstraintFunction constraintFunTest3("testConstraintFunction3",4);
	constraintFunTest3.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest3.setID(2);
	constraintFunTest3.setGradientOn();
	constraintFunTest3.setReadMarker("Constraint3");
	constraintFunTest3.setReadMarkerAdjoint("Constraint3Gradient");

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);

	rowvec gradientResult = d.constraintGradients[0];
	EXPECT_EQ(gradientResult(0),0);
	EXPECT_EQ(gradientResult(1),0);
	EXPECT_EQ(gradientResult(2),0);
	EXPECT_EQ(gradientResult(3),0);



	constraintFunTest2.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(1),-1);

	gradientResult = d.constraintGradients[1];
	EXPECT_EQ(gradientResult(0),1);
	EXPECT_EQ(gradientResult(1),2);
	EXPECT_EQ(gradientResult(2),3);
	EXPECT_EQ(gradientResult(3),4);

	constraintFunTest3.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(2),-1.6);

	gradientResult = d.constraintGradients[2];
	EXPECT_EQ(gradientResult(0),-1);
	EXPECT_EQ(gradientResult(1),-2);
	EXPECT_EQ(gradientResult(2),-3);
	EXPECT_EQ(gradientResult(3),-4);

	remove("readOutputTestFile.txt");
}



TEST(testConstraintFunctions, testreadEvaluateObjectiveFunctionAndConstraintInTheSameFile){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "Constraint1 = 2.144\nObjectiveFunction = 1.789";
	readOutputTestFile.close();


	/* constraint has only value */
	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setReadMarker("Constraint1");
	constraintFunTest.setID(0);

	/* objective function has only value */
	ObjectiveFunction objFunTest("testObjFun",4);
	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setReadMarker("ObjectiveFunction");


	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);


	objFunTest.readEvaluateOutput(d);

	EXPECT_EQ(d.trueValue,1.789);
	EXPECT_EQ(d.objectiveFunctionValue,1.789);

	remove("readOutputTestFile.txt");

}

TEST(testConstraintFunctions, testreadEvaluateObjectiveFunctionAndConstraintInTheSameFileWithAdjoints){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "Constraint1 = 2.144\nConstraint1Gradient = 1.2,2.2,3.1,4.1\nObjectiveFunction = 1.789\nObjectiveFunctionGradient = 0,1,2,3";
	readOutputTestFile.close();

	ObjectiveFunction objFunTest("testObjFun",4);
	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setReadMarker("ObjectiveFunction");
	objFunTest.setReadMarkerAdjoint("ObjectiveFunctionGradient");
	objFunTest.setGradientOn();

	objFunTest.readEvaluateOutput(d);

	EXPECT_EQ(d.trueValue,1.789);

	rowvec gradientResult = d.gradient;
	EXPECT_EQ(gradientResult(0),0);
	EXPECT_EQ(gradientResult(1),1);
	EXPECT_EQ(gradientResult(2),2);
	EXPECT_EQ(gradientResult(3),3);


	ConstraintFunction constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setReadMarker("Constraint1");
	constraintFunTest.setReadMarkerAdjoint("Constraint1Gradient");
	constraintFunTest.setGradientOn();
	constraintFunTest.setID(0);

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);

	gradientResult = d.constraintGradients[0];
	EXPECT_EQ(gradientResult(0),1.2);
	EXPECT_EQ(gradientResult(1),2.2);
	EXPECT_EQ(gradientResult(2),3.1);
	EXPECT_EQ(gradientResult(3),4.1);

	remove("readOutputTestFile.txt");


}

#endif
