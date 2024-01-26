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
 
#include "../INCLUDE/matrix_operations.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../../Random/INCLUDE/random_functions.hpp"
#include "../../Bounds/INCLUDE/bounds.hpp"

 
#include<math.h>
#include<gtest/gtest.h>



TEST(testMatrixOperations, isEqualRowVector){

	rowvec v1(5,fill::randu);
	rowvec v3(5,fill::randu);
	rowvec v2 = v1;
	v2(0) += 10E-8;

	ASSERT_TRUE(isEqual(v1,v2,10E-5));
	ASSERT_FALSE(isEqual(v1,v2,10E-9));

}



TEST(testMatrixOperations, findIndexOfRow){

	mat m1(5,5,fill::randu);
	rowvec v1 = m1.row(2);
	rowvec v2(5, fill::randu);

	ASSERT_TRUE(findIndexOfRow(v1, m1,10E-08) == 2 );
	ASSERT_TRUE(findIndexOfRow(v2, m1,10E-08) == -1 );


}

TEST(testMatrixOperations, findIndexOfRowVeryCloseSamples){

	mat m1(5,5,fill::randu);
	rowvec v = m1.row(2);
	rowvec w = v + 10E-06;
	m1.row(0) = w;

	ASSERT_TRUE(findIndexOfRow(v, m1,10E-04) == 2 );


}




TEST(testMatrixOperations, normalizeMatrix){


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


TEST(testMatrixOperations, joinMatricesByColumns){

	mat A(4,2,fill::randu);
	mat B(4,5,fill::randu);

	joinMatricesByColumns(A,B);

	ASSERT_TRUE(A.n_cols == 7);
	double error = fabs(B(1,2) - A(1,4));
	EXPECT_LT(error,10E-10);


}

TEST(testMatrixOperations, joinMatricesByRows){

	mat A(4,6,fill::randu);
	mat B(3,6,fill::randu);

	joinMatricesByRows(A,B);

	ASSERT_TRUE(A.n_rows == 7);
	double error = fabs(B(1,2) - A(5,2));
	EXPECT_LT(error,10E-10);


}

TEST(testMatrixOperations, checkifTooCLoseVectorMatrixVersion){

	rowvec x1(3, fill::randu);
	mat M(20,3,fill::randu);
	M.row(5) = x1;
	bool check = checkifTooCLose(x1,M, 10E-08);
	ASSERT_TRUE(check);

	M.row(5) = 2*x1;
	check = checkifTooCLose(x1,M,10E-08);
	ASSERT_FALSE(check);

}


