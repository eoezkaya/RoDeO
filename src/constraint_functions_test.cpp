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

TEST(testConstraintFunctions, testsetInequalityTypeDoesWork){

	ConstraintFunctionv2 constraintFunTest("testConstraintFunction",Himmelblau,2);
	constraintFunTest.setInequalityType("lt");
	constraintFunTest.setInequalityType("gt");

}

