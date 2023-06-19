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



#include "../INCLUDE/general_purpose_optimizer.hpp"
#include "../../TestFunctions/INCLUDE/standard_test_functions.hpp"
#include<gtest/gtest.h>


class GeneralPurposeOptimizerTest : public ::testing::Test {
protected:
	void SetUp() override {






	}

	//  void TearDown() override {}

	GeneralPurposeOptimizer testOptimizer;


};
//
//
//TEST_F(GeneralPurposeOptimizerTest, testIfConstructorWorks) {
//
//	ASSERT_TRUE(testOptimizer.getDimension() == 0);
//	ASSERT_FALSE(testOptimizer.areBoundsSet());
//
//
//}
//
//TEST_F(GeneralPurposeOptimizerTest, testIfsetBoundsWorksWithDoubles) {
//
//	testOptimizer.setDimension(2);
//	testOptimizer.setBounds(0.0,1.0);
//	ASSERT_TRUE(testOptimizer.areBoundsSet());
//
//
//}
//
//TEST_F(GeneralPurposeOptimizerTest, testIfsetBoundsWorksWithVectors) {
//
//	vec lb(2,fill::zeros);
//	vec ub(2,fill::ones);
//	testOptimizer.setDimension(2);
//	testOptimizer.setBounds(lb,ub);
//	ASSERT_TRUE(testOptimizer.areBoundsSet());
//
//
//}
//
//TEST_F(GeneralPurposeOptimizerTest, testIfsetBoundsWorksWithBoundsClass) {
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,1.0);
//	testOptimizer.setDimension(2);
//	testOptimizer.setBounds(testBounds);
//	ASSERT_TRUE(testOptimizer.areBoundsSet());
//
//
//}
//
//TEST_F(GeneralPurposeOptimizerTest, testsetObjectiveFunction) {
//
//
//	testOptimizer.setObjectiveFunction(Eggholder);
//	ASSERT_TRUE(testOptimizer.isObjectiveFunctionSet());
//
//
//}
//
//TEST_F(GeneralPurposeOptimizerTest, testcallObjectiveFunction){
//
//	vec x(2); x(0) = 100.0; x(1) = 100.0;
//	testOptimizer.setObjectiveFunction(Eggholder);
//
//
//	double value = testOptimizer.callObjectiveFunction(x);
//	double valueToCheck = Eggholder(x);
//	double error = fabs(value - valueToCheck);
//
//	ASSERT_LT(error,10E-10);
//
//}
//TEST_F(GeneralPurposeOptimizerTest, testcallObjectiveFunctionInternal){
//
//	vec x(2); x(0) = 100.0; x(1) = 100.0;
//
//
//	double value = testOptimizer.callObjectiveFunction(x);
//	double valueToCheck = -19.12;
//	double error = fabs(value - valueToCheck);
//
//	ASSERT_LT(error,10E-10);
//
//}


