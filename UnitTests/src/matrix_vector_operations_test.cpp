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
#include<math.h>
#include "matrix_vector_operations.hpp"
#include "auxiliary_functions.hpp"
#include "bounds.hpp"
#include "random_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>


#ifdef TEST_MATRIX_VECTOR_OPS


TEST(testMatrixVectorOperations, isEqualRowVector){

	rowvec v1(5,fill::randu);
	rowvec v3(5,fill::randu);
	rowvec v2 = v1;
	v2(0) += 10E-8;

	ASSERT_TRUE(isEqual(v1,v2,10E-5));
	ASSERT_FALSE(isEqual(v1,v2,10E-9));

}

TEST(testMatrixVectorOperations, isEqualMatrix){

	mat m1(5,5,fill::randu);
	mat m2(5,5,fill::randu);


	ASSERT_FALSE(isEqual(m1,m2,10E-5));

	m2 = m1;
	m2 +=10E-6;

	ASSERT_FALSE(isEqual(m1,m2,10E-8));
	ASSERT_TRUE(isEqual(m1,m2,10E-4));


}



TEST(testMatrixVectorOperations, findIndexOfRow){

	mat m1(5,5,fill::randu);
	rowvec v1 = m1.row(2);
	rowvec v2(5, fill::randu);

	ASSERT_TRUE(findIndexOfRow(v1, m1,10E-08) == 2 );
	ASSERT_TRUE(findIndexOfRow(v2, m1,10E-08) == -1 );


}


TEST(testMatrixVectorOperations, abortIfHasNan){

	unsigned int dim = 5;
	rowvec testVector = zeros<rowvec>(dim);

	//	testVector(0) = sqrt(-1);

	abortIfHasNan(testVector);


}


TEST(testMatrixVectorOperations, copyRowVector){

	rowvec a(10,fill::randu);
	rowvec b(5, fill::randu);

	copyRowVector(a,b);

	EXPECT_EQ(a(0), b(0));
	EXPECT_EQ(a(1), b(1));
	EXPECT_EQ(a(2), b(2));
	EXPECT_EQ(a(3), b(3));
	EXPECT_EQ(a(4), b(4));

}


TEST(testMatrixVectorOperations, normalizeRowVector){

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


TEST(testMatrixVectorOperations, normalizeMatrix){


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

TEST(testMatrixVectorOperations, findInterval){

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
TEST(testMatrixVectorOperations, convertToVector){

	rowvec testRowVector(10,fill::randu);

	vec testVector = convertToVector(testRowVector);
	ASSERT_TRUE(testVector.size() == testRowVector.size());

	double error  = fabs(testRowVector(5) - testVector(5));

	ASSERT_LT(error, 10E-10);


}
TEST(testMatrixVectorOperations, convertToRowVector){

	vec testVector(10,fill::randu);

	rowvec testRowVector = convertToRowVector(testVector);
	ASSERT_TRUE(testVector.size() == testRowVector.size());

	double error  = fabs(testRowVector(5) - testVector(5));

	ASSERT_LT(error, 10E-10);


}
TEST(testMatrixVectorOperations, addOneElement){

	vec testVector(10,fill::randu);
	double val = testVector(5);

	double someValue = 1.987;
	addOneElement(testVector, someValue);
	ASSERT_TRUE(testVector.size() == 11);
	double error  = fabs(testVector(5) - val);
	ASSERT_LT(error, 10E-10);
	error  = fabs(testVector(10) - someValue);
	ASSERT_LT(error, 10E-10);

	vec testRowVector(10,fill::randu);
	val = testRowVector(5);
	addOneElement(testRowVector, someValue);
	ASSERT_TRUE(testRowVector.size() == 11);

	error  = fabs(testRowVector(5) - val);
	ASSERT_LT(error, 10E-10);
	error  = fabs(testRowVector(10) - someValue);
	ASSERT_LT(error, 10E-10);


}

TEST(testMatrixVectorOperations, joinMatricesByColumns){

	mat A(4,2,fill::randu);
	mat B(4,5,fill::randu);

	joinMatricesByColumns(A,B);

	ASSERT_TRUE(A.n_cols == 7);
	double error = fabs(B(1,2) - A(1,4));
	EXPECT_LT(error,10E-10);


}

TEST(testMatrixVectorOperations, joinMatricesByRows){

	mat A(4,6,fill::randu);
	mat B(3,6,fill::randu);

	joinMatricesByRows(A,B);

	ASSERT_TRUE(A.n_rows == 7);
	double error = fabs(B(1,2) - A(5,2));
	EXPECT_LT(error,10E-10);


}

TEST(testMatrixVectorOperations, testMakeUnitVector){

	vec v(10,fill::randu);
	v = v*5;
	vec w = makeUnitVector(v);
	double normw = norm(w,2);
	EXPECT_LT(fabs(normw-1.0),10E-10);
}

TEST(testMatrixVectorOperations, testMakeUnitRowVector){

	rowvec v(10,fill::randu);
	v = v*5;
	rowvec w = makeUnitVector(v);

	double normw = norm(w,2);
	EXPECT_LT(fabs(normw-1.0),10E-10);

}

TEST(testMatrixVectorOperations, isIntheList){

	uvec list(4);
	list(0) = 1;
	list(1) = 6;
	list(2) = 3;
	list(3) = 10;

	ASSERT_TRUE(isIntheList(list,10));

}

TEST(testMatrixVectorOperations, isIntheListFalse){

	uvec list(4);
	list(0) = 1;
	list(1) = 6;
	list(2) = 3;
	list(3) = 10;

	ASSERT_FALSE(isIntheList(list,11));

}

TEST(testMatrixVectorOperations, isIntheListStdVector){

	vector<int> list(4);
	list[0] = 1;
	list[1] = 6;
	list[2] = 3;
	list[3] = 10;

	ASSERT_TRUE(isIntheList(list,10));

}

TEST(testMatrixVectorOperations, isIntheListStdVectorFalse){

	vector<int> list(4);
	list[0] = 1;
	list[1] = 6;
	list[2] = 3;
	list[3] = 10;

	ASSERT_FALSE(isIntheList(list,11));

}


TEST(testMatrixVectorOperations, returnKMinIndices){

	vec v(6);
	v(0) = 1.9; v(1) = -1.9; v(2) = 5.23; v(3) = 8.9; v(4) = 11.9; v(5) = 1.9;

	vector<int> kBest = returnKMinIndices(v,3);

	ASSERT_TRUE(isIntheList(kBest,1));
	ASSERT_TRUE(isIntheList(kBest,0));
	ASSERT_TRUE(isIntheList(kBest,5));
	ASSERT_FALSE(isIntheList(kBest,2));
	ASSERT_FALSE(isIntheList(kBest,3));
	ASSERT_FALSE(isIntheList(kBest,4));

}


TEST(testMatrixVectorOperations, returnKMinIndices2){

	vec v(10);
	v(0) =  0.0;
	v(1) = -12.9;
	v(2) =  5.23;
	v(3) =  8.9;
	v(4) = 11.9;
	v(5) = 1.912;
	v(6) = 2.912;
	v(7) = 3.912;
	v(8) = 4.912;
	v(9) = -5.912;

	vector<int> kBest = returnKMinIndices(v,2);

	ASSERT_TRUE(isIntheList(kBest,1));
	ASSERT_TRUE(isIntheList(kBest,9));
	ASSERT_FALSE(isIntheList(kBest,0));
	ASSERT_TRUE(kBest.size() == 2);


}


TEST(testMatrixVectorOperations, returnKMinIndices3){

	vec v(10);
	v(0) =  0.0;
	v(1) =  1.0;
	v(2) =  2.0;
	v(3) =  3.0;
	v(4) =  4.0;
	v(5) =  5.0;
	v(6) =  0.0;
	v(7) =  1.0;
	v(8) =  2.9;
	v(9) =  100.0;

	vector<int> kBest = returnKMinIndices(v,4);

	ASSERT_TRUE(isIntheList(kBest,0));
	ASSERT_TRUE(isIntheList(kBest,1));
	ASSERT_TRUE(isIntheList(kBest,6));
	ASSERT_FALSE(isIntheList(kBest,9));
	ASSERT_TRUE(kBest.size() == 4);


}


TEST(testMatrixVectorOperations, findIndicesKMax){

	vec v(6);
	v(0) = 1.9; v(1) = -1.9; v(2) = 5.23; v(3) = 8.9; v(4) = 11.9; v(5) = 1.9;

	uvec kBest = findIndicesKMax(v,3);
	ASSERT_TRUE(isIntheList(kBest,4));


}


TEST(testMatrixVectorOperations, findIndicesKMin){

	vec v(6);
	v(0) = 1.9; v(1) = -1.9; v(2) = 5.23; v(3) = 8.9; v(4) = 11.9; v(5) = 1.9;

	uvec kBest = findIndicesKMin(v,3);
	ASSERT_TRUE(isIntheList(kBest,1));

}



TEST(testMatrixVectorOperations, shuffleRows){

	mat A(6,5,fill::randu);
	A = shuffleRows(A);

	ASSERT_TRUE(A.n_rows == 6);
	ASSERT_TRUE(A.n_cols == 5);
}


TEST(testMatrixVectorOperations, isBetweenRowVec){

	rowvec v1(3);
	rowvec lb(3);
	rowvec ub(3);

	v1(0) = 0.2;  v1(1) = 0.8; v1(2) = 0.9;
	lb(0) = 0.1;  lb(1) = 0.0; lb(2) = -1.9;
	ub(0) = 0.3;  ub(1) = 1.0; ub(2) = 1.9;

	bool ifBetween = isBetween(v1,lb,ub);

	ASSERT_TRUE(ifBetween);

	v1(0) = 0.1;
	ifBetween = isBetween(v1,lb,ub);

	ASSERT_TRUE(ifBetween);

	v1(0) = -0.1;
	ifBetween = isBetween(v1,lb,ub);

	ASSERT_FALSE(ifBetween);
}

#endif
