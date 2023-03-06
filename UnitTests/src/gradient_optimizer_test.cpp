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


#include "gradient_optimizer.hpp"
#include "test_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>


#ifdef  TEST_GRADIENTOPTIMIZER


class GradientOptimizerTest : public ::testing::Test {
protected:
	void SetUp() override {

		testOptimizer.setDimension(2);
		testOptimizer.setBounds(0.0,200.0);
		testOptimizer.setObjectiveFunction(Himmelblau);
		testOptimizer.setGradientFunction(HimmelblauGradient);




	}

	//  void TearDown() override {}

	GradientOptimizer testOptimizer;


};

TEST_F(GradientOptimizerTest, testIfConstructorWorks) {

	ASSERT_TRUE(testOptimizer.getDimension() == 2);
	ASSERT_TRUE(testOptimizer.areBoundsSet());
	ASSERT_TRUE(testOptimizer.isObjectiveFunctionSet());
	ASSERT_FALSE(testOptimizer.isInitialPointSet());

}

TEST_F(GradientOptimizerTest, testSetInitialPoint) {

	vec x0(2);
	x0(0) = 1.0; x0(1) = 0.0;
	testOptimizer.setInitialPoint(x0);
	ASSERT_TRUE(testOptimizer.isInitialPointSet());


}

TEST_F(GradientOptimizerTest, testEvaluateObjectiveFunction) {

	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	testOptimizer.evaluateObjectiveFunction(iterateTest);

	double error = fabs(iterateTest.objectiveFunctionValue - Himmelblau(x));

	EXPECT_LT(error, 10E-08);


}



TEST_F(GradientOptimizerTest, testEvaluateGradientFunction) {

	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	ASSERT_TRUE(testOptimizer.isGradientFunctionSet());

	testOptimizer.evaluateGradientFunction(iterateTest);

	vec gradient = HimmelblauGradient(x);

	double errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	double errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-08);
	EXPECT_LT(errorGradient2, 10E-08);

}

TEST_F(GradientOptimizerTest, testEvaluateGradientFunctionWithFiniteDifference) {


	vec x(2);
	x(0) = 1.7; x(1) = 1.2;

	designPoint iterateTest;
	iterateTest.x = x;

	testOptimizer.setFiniteDifferenceMethod("central");
	testOptimizer.evaluateGradientFunction(iterateTest);


	vec gradient = HimmelblauGradient(x);


	double errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	double errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-05);
	EXPECT_LT(errorGradient2, 10E-05);


	testOptimizer.setFiniteDifferenceMethod("forward");
	testOptimizer.evaluateGradientFunction(iterateTest);


	errorGradient1 = fabs(iterateTest.gradient(0) - gradient(0));
	errorGradient2 = fabs(iterateTest.gradient(1) - gradient(1));

	EXPECT_LT(errorGradient1, 10E-05);
	EXPECT_LT(errorGradient2, 10E-05);
}

TEST_F(GradientOptimizerTest, testOptimizeWithSteepestDescent) {


	testOptimizer.setMaximumNumberOfFunctionEvaluations(200);
	testOptimizer.setMaximumStepSize(1.0);

	Bounds boxConstraints(2);
	boxConstraints.setBounds(-6.0,6.0);

	vec x0(2);
	x0(0) = 1.7; x0(1) = 1.2;
	testOptimizer.setInitialPoint(x0);

	testOptimizer.setBounds(boxConstraints);
	testOptimizer.optimize();



	double J = testOptimizer.getOptimalObjectiveFunctionValue();

	double error = fabs(J - 0.0);

	EXPECT_LT(error, 10E-06);


}



#endif
