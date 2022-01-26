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
#include<math.h>
#include "matrix_vector_operations.hpp"
#include "bounds.hpp"
#include "random_functions.hpp"
#include<gtest/gtest.h>


TEST(testMatrixVectorOperations, testabortIfHasNan){

	unsigned int dim = 5;
	rowvec testVector = zeros<rowvec>(dim);

//	testVector(0) = sqrt(-1);

	abortIfHasNan(testVector);


}


TEST(testMatrixVectorOperations, testcopyRowVector){

	rowvec a(10,fill::randu);
	rowvec b(5, fill::randu);

	copyRowVector(a,b);

	EXPECT_EQ(a(0), b(0));
	EXPECT_EQ(a(1), b(1));
	EXPECT_EQ(a(2), b(2));
	EXPECT_EQ(a(3), b(3));
	EXPECT_EQ(a(4), b(4));

}


TEST(testMatrixVectorOperations, testnormalizeRowVector){

	rowvec x(5,fill::randu);
	vec xmin(5); xmin.fill(0.1);
	vec xmax(5); xmax.fill(2.1);

	rowvec xNormalized = normalizeRowVector(x, xmin, xmax);
	double xCheck = xNormalized(0)*5 * (2.0) + 0.1;

	double error = fabs(xCheck-x(0));
	EXPECT_LT(error, 10E-10);

	x(0) = 1.3; x(1) = 10.7; x(2) = -1.3; x(3) = 0.0; x(4) = 1.7;
	xmin.fill(1.3);
	xmax.fill(50.0);
	xNormalized = normalizeRowVector(x, xmin, xmax);
	EXPECT_LT(fabs(xNormalized(0)), 10E-10);

}


TEST(testMatrixVectorOperations, testnormalizeMatrix){


	unsigned int dim1 = 10;
	unsigned int dim2 = 57;

	mat testData(dim2,dim1, fill::randu);

	vec lb(dim1-1, fill::randu);
	vec ub(dim1-1, fill::randu);

	ub = ub + 1.0;

	Bounds boxConstraints(lb,ub);

	mat testDataNormalized = normalizeMatrix(testData, boxConstraints);

	double expectedValue = (testData(0,0) - lb(0))/(ub(0) - lb(0));

	EXPECT_EQ(expectedValue,testDataNormalized(0,0));
	EXPECT_EQ(testData(0,dim1-1),testDataNormalized(0,dim1-1));


}

TEST(testMatrixVectorOperations, testfindInterval){

	vec discreteValues(5);
	discreteValues(0) = -1.8;
	discreteValues(1) = -1.6;
	discreteValues(2) = 0;
	discreteValues(3) = 1.0;
	discreteValues(4) = 190;

	int index = findInterval(0.9, discreteValues);
	EXPECT_EQ(index,2);

	index = findInterval(-1.9, discreteValues);
	EXPECT_EQ(index,-1);

}

