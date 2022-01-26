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


#include "lhs.hpp"
#include<gtest/gtest.h>

#define TEST_LHS_ON

#ifdef TEST_LHS_ON

TEST(testLHS, testLHSConstructor){

	unsigned int dim = 4;
	unsigned int N = 10;
	vec lb(dim);
	vec ub(dim);
	lb.fill(0.0);
	ub.fill(1.0);

	LHSSamples LHSTest(dim, lb, ub, N);

	mat samples = LHSTest.getSamples();

	EXPECT_EQ(samples.n_rows,N);
	EXPECT_EQ(samples.n_cols,dim);


}


TEST(testLHS, testcalculateDiscreteParameterValues){

	unsigned int dim = 4;
	unsigned int N = 10;
	vec lb(dim);
	vec ub(dim);
	lb.fill(0.0);
	ub.fill(1.0);

	LHSSamples LHSTest(dim, lb, ub, N);

	mat samples = LHSTest.getSamples();

	int indices[2] = {0,2};
	vec increments(2);
	increments(0) = 0.1;
	increments(1) = 0.25;


	LHSTest.printSamples();
	LHSTest.setDiscreteParameterIndices(indices,2);
	LHSTest.setDiscreteParameterIncrements(increments);
	LHSTest.roundSamplesToDiscreteValues();
	LHSTest.printSamples();


}



#endif
