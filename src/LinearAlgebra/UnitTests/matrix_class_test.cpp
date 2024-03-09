/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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

#include "../INCLUDE/matrix.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"


#include<gtest/gtest.h>

using namespace rodeo;

rodeo::mat testFunction(rodeo::mat A){

	return A;
}

class MatrixClassTest : public ::testing::Test {
protected:
	void SetUp() override {


	}

	//  void TearDown() override {}


};

TEST_F(MatrixClassTest, constructor) {

	rodeo::mat test(2,3);

	ASSERT_TRUE(test.getSize()  == 6);
	ASSERT_TRUE(test.getNCols() == 3);
	ASSERT_TRUE(test.getNRows() == 2);
}
TEST_F(MatrixClassTest, accessOperator) {

	rodeo::mat test(2,3);
	test(1,2) = 5;
	//	test.print();


	double a = test(1,2);

	ASSERT_TRUE(fabs(a-5)<10E-10);

}

TEST_F(MatrixClassTest, callFunction) {

	rodeo::mat test(100,100);

	rodeo::mat test2 = testFunction(test);
	rodeo::mat test3 = test2;

}

TEST_F(MatrixClassTest, eye) {

	rodeo::mat test = rodeo::mat::eye(5);

}

TEST_F(MatrixClassTest, fill) {

	rodeo::mat test = rodeo::mat::eye(5);
	test.fill(-1.12);
}

TEST_F(MatrixClassTest, fillRandom) {

	rodeo::mat test = rodeo::mat::eye(5);
	test.fillRandom();
}

TEST_F(MatrixClassTest, multiplyWithScalar) {

	rodeo::mat test(5,3);
	test.fill(-1.0);
	rodeo::mat test2 = test*1.13;

}

TEST_F(MatrixClassTest, resizeSmaller) {

	rodeo::mat test(6,5, 2.33);
	ASSERT_TRUE(test.getNCols() == 5);
	ASSERT_TRUE(test.getNRows() == 6);

	test.resize(2,3);
	ASSERT_TRUE(test.getNCols() == 3);
	ASSERT_TRUE(test.getNRows() == 2);

}

TEST_F(MatrixClassTest, resizeLarger) {

	rodeo::mat test(6,5, 2.33);
	ASSERT_TRUE(test.getNCols() == 5);
	ASSERT_TRUE(test.getNRows() == 6);

	test.resize(2,7);
	ASSERT_TRUE(test.getNCols() == 7);
	ASSERT_TRUE(test.getNRows() == 2);

}

TEST_F(MatrixClassTest, subview) {

	rodeo::mat test(6,5, 2.33);
	test.fillRandom();
	//	test.print();

	rodeo::mat test2 = test.submat(2,1,2,2);
	//	test2.print();

}

TEST_F(MatrixClassTest, addColumns) {


	rodeo::mat A(3, 2);
	// Add 2 columns to the matrix with a default value of 1.0
	A.addColumns(2,1.0);

	ASSERT_TRUE(A.getNCols() == 4);
	ASSERT_TRUE(A.getNRows() == 3);
	ASSERT_TRUE(A(2,3) == 1.0);

}

TEST_F(MatrixClassTest, concatenateRowWise) {

	rodeo::mat A(2, 3);
	rodeo::mat B(3, 3);

	A.fill(1.8);
	B.fill(1.9);


	// Concatenate matrices A and B row-wise
	rodeo::mat C = A.concatenateRowWise(B);


	ASSERT_TRUE(C.getNCols() == 3);
	ASSERT_TRUE(C.getNRows() == 5);

}

TEST_F(MatrixClassTest, concatenateColumnWise) {

	rodeo::mat A(2, 3);
	rodeo::mat B(2, 2);
	A.fillRandom();
	B.fillRandom();

	// Concatenate matrices A and B column-wise
	rodeo::mat C = A.concatenateColumnWise(B);

	ASSERT_TRUE(C.getNCols() == 5);
	ASSERT_TRUE(C.getNRows() == 2);

}



