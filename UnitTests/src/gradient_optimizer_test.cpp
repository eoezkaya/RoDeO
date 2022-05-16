/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022. Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


#include "gradient_optimizer.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>

#define TESTGRADIENTOPTIMIZER
#ifdef  TESTGRADIENTOPTIMIZER

TEST(testGradientOptimizer, testGradientOptimizer_Constructor){

	GradientOptimizer test;

	ASSERT_TRUE(test.isOptimizationTypeMinimization());
	ASSERT_FALSE(test.areBoundsSet());
	ASSERT_FALSE(test.isInitialPointSet());
	ASSERT_FALSE(test.isObjectiveFunctionSet());



}


TEST(testGradientOptimizer, testGradientOptimizer_setInitialPoint){

	GradientOptimizer test;

	vec x0(3);
	x0(0) = 1.0; x0(1) = 0.0; x0(2) = -1.0;
	test.setDimension(3);
	test.setInitialPoint(x0);

	ASSERT_TRUE(test.isInitialPointSet());

}

TEST(testGradientOptimizer, testGradientOptimizer_setObjectiveFunction){

	GradientOptimizer test;

	test.setObjectiveFunction(Eggholder);

	ASSERT_TRUE(test.isObjectiveFunctionSet());


}


TEST(testGradientOptimizer, testGradientOptimizer_evaluateObjectiveFunction){

	GradientOptimizer test;

	test.setObjectiveFunction(Eggholder);

	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	test.evaluateObjectiveFunction(iterateTest);

	double error = fabs(iterateTest.objectiveFunctionValue - Eggholder(x));

	EXPECT_LT(error, 10E-08);


}





TEST(testGradientOptimizer, testGradientOptimizer_setGradientFunction){

	GradientOptimizer test;

	test.setGradientFunction(HimmelblauGradient);

	ASSERT_TRUE(test.isGradientFunctionSet());



}




TEST(testGradientOptimizer, testGradientOptimizer_evaluateGradientFunction){

	GradientOptimizer test;

	test.setGradientFunction(HimmelblauGradient);

	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	test.evaluateGradientFunction(iterateTest);

	vec gradient = HimmelblauGradient(x);

	double errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	double errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-08);
	EXPECT_LT(errorGradient2, 10E-08);



}


TEST(testGradientOptimizer, testGradientOptimizer_evaluateGradientFunctionWithFiniteDifference){

	GradientOptimizer test;
	test.setDimension(2);
	test.setObjectiveFunction(Himmelblau);

	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	test.setFiniteDifferenceMethod("central");
	test.evaluateGradientFunction(iterateTest);


	vec gradient = HimmelblauGradient(x);

	double errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	double errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-05);
	EXPECT_LT(errorGradient2, 10E-05);


	test.setFiniteDifferenceMethod("forward");
	test.evaluateGradientFunction(iterateTest);
	errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-05);
	EXPECT_LT(errorGradient2, 10E-05);

}


TEST(testGradientOptimizer, testGradientOptimizer_optimizeWithSteepestDescent){

	GradientOptimizer test;

	test.setObjectiveFunction(Himmelblau);
	test.setGradientFunction(HimmelblauGradient);
	test.setDimension(2);

	vec x0(2);
	x0(0) = 1.7; x0(1) = 1.2;
	test.setInitialPoint(x0);

	Bounds boxConstraints(2);
	boxConstraints.setBounds(-6.0,6.0);

	test.setMaximumNumberOfFunctionEvaluations(200);
	test.setMaximumStepSize(1.0);
	test.setBounds(boxConstraints);
	//	test.setDisplayOn();

	test.optimize();



	double J = test.getOptimalObjectiveFunctionValue();

	double error = fabs(J - 0.0);

	EXPECT_LT(error, 10E-06);


}


TEST(testGradientOptimizer, testGradientOptimizer_optimizeWithSteepestDescentWithFiniteDifferences){

	GradientOptimizer test;

	test.setObjectiveFunction(Himmelblau);
	test.setFiniteDifferenceMethod("central");
	test.setDimension(2);

	vec x0(2);
	x0(0) = 1.7; x0(1) = 1.2;
	test.setInitialPoint(x0);

	Bounds boxConstraints(2);
	boxConstraints.setBounds(-6.0,6.0);

	test.setMaximumNumberOfFunctionEvaluations(200);
	test.setMaximumStepSize(1.0);
	test.setBounds(boxConstraints);
	//	test.setDisplayOn();
	test.setEpsilonForFiniteDifference(0.000001);

	test.optimize();



	double J = test.getOptimalObjectiveFunctionValue();

	double error = fabs(J - 0.0);

	EXPECT_LT(error, 10E-06);


}

#endif
