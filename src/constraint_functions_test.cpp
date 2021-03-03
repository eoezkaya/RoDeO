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

#include<gtest/gtest.h>
#include "constraint_functions.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include "design.hpp"

TEST(testConstraintFunctions, testConstructor){

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",4);
	unsigned int dim = constraintFunTest.getDimension();
	std::string name = constraintFunTest.getName();

	ASSERT_EQ(dim,4);
	ASSERT_TRUE(name.compare("testConstraintFunction") == 0);


}

TEST(testConstraintFunctions, testConstructorWithFunctionPointer){

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",Himmelblau,2);


	unsigned int dim = constraintFunTest.getDimension();
	std::string name = constraintFunTest.getName();

	ASSERT_EQ(dim,2);
	ASSERT_TRUE(name.compare("testConstraintFunction") == 0);
	ASSERT_TRUE(constraintFunTest.ifHasFunctionFunctionPointer());


}

TEST(testConstraintFunctions, testsetInequalityConstraint){

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",Himmelblau,2);
	constraintFunTest.setInequalityConstraint(" > 18.2");


}

TEST(testConstraintFunctions, checkFeasibility){

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",4);

	constraintFunTest.setInequalityConstraint(" > 18.2");
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


	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",2);
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

TEST(testConstraintFunctions, testreadEvaluateOutput){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144\n";
	readOutputTestFile.close();

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(1);

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);

}

TEST(testConstraintFunctions, testreadEvaluateOutputWithGradient){

	Design d(4);
	d.setNumberOfConstraints(1);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 -1 -2 -3 -4\n";
	readOutputTestFile.close();

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(1);
	constraintFunTest.setGradientOn();

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);

	rowvec gradientResult = d.constraintGradients.front();
	EXPECT_EQ(gradientResult(0),-1);
	EXPECT_EQ(gradientResult(1),-2);
	EXPECT_EQ(gradientResult(2),-3);
	EXPECT_EQ(gradientResult(3),-4);
}

TEST(testConstraintFunctions, testreadEvaluateTwoOutputsInTheSameFile){

	Design d(4);
	d.setNumberOfConstraints(2);
	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 -1\n";
	readOutputTestFile.close();

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",4);
	constraintFunTest.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest.setID(1);

	ConstraintFunctionv2 constraintFunTest2("testConstraintFunction2",4);
	constraintFunTest2.setFileNameReadInput("readOutputTestFile.txt");
	constraintFunTest2.setID(2);
	constraintFunTest2.readOutputStartIndex = 1;

	constraintFunTest.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(0),2.144);

	constraintFunTest2.readEvaluateOutput(d);
	EXPECT_EQ(d.constraintTrueValues(1),-1);

}






