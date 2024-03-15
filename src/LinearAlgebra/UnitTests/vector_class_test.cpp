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

#include "../INCLUDE/vector.hpp"
#include "../INCLUDE/matrix.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"


#include<gtest/gtest.h>

using namespace rodeo;

rodeo::vec testFunction(rodeo::vec v){

	return v;
}


class VectorClassTest : public ::testing::Test {
protected:

	void SetUp() override {}

	rodeo::vec vector;
	rodeo::mat matrix;
};



TEST_F(VectorClassTest, constructor) {

	rodeo::vec test(3);
	ASSERT_TRUE(rodeo::vec::len(test) == 3);
}

TEST_F(VectorClassTest, accessOperator) {

	rodeo::vec test(3);
	test(0) = 3.3;
	ASSERT_TRUE(fabs(test(0) - 3.3) < 10E-10);
}

TEST_F(VectorClassTest, fill) {

	rodeo::vec test(3);
	test.fill(3.3);
	ASSERT_TRUE(fabs(test(0) - 3.3) < 10E-10);
}

TEST_F(VectorClassTest, resize) {

	rodeo::vec test(3);
	test.fill(3.3);
	test.resize(5);
	ASSERT_TRUE(fabs(test(4) - 0.0) < 10E-10);

}

TEST_F(VectorClassTest, resizeSmaller) {

	rodeo::vec test(10);
	test.fill(3.3);
	test.resize(5);
	ASSERT_TRUE(test.getSize() == 5);
}

TEST_F(VectorClassTest, fillRandom) {

	rodeo::vec test(10);
	test.fillRandom();
}

TEST_F(VectorClassTest,isZero) {

	rodeo::vec v(3);
	ASSERT_TRUE(v.isZero());
	v(1) = 0.1;
	ASSERT_FALSE(v.isZero());
}


TEST_F(VectorClassTest, calculateL1norm) {

	rodeo::vec v(3);
	v(0) = -2.0;
	v(1) = 3.0;
	v(2) = -1.0;

	double norm = v.calculateL1Norm();
	ASSERT_TRUE(fabs(norm - 6.0) < 10E-10);
}

TEST_F(VectorClassTest, head) {

	rodeo::vec v(7);
	v.fillRandom();

	rodeo::vec w = v.head(5);

	ASSERT_TRUE(w.getSize() == 5);
	ASSERT_TRUE(fabs(w(0) - v(0)) < 10E-10);
	ASSERT_TRUE(fabs(w(1) - v(1)) < 10E-10);
	ASSERT_TRUE(fabs(w(2) - v(2)) < 10E-10);
	ASSERT_TRUE(fabs(w(3) - v(3)) < 10E-10);
	ASSERT_TRUE(fabs(w(4) - v(4)) < 10E-10);
}

TEST_F(VectorClassTest, tail) {

	rodeo::vec v(7);
	v.fillRandom();

	rodeo::vec w = v.tail(5);

	ASSERT_TRUE(w.getSize() == 5);
	ASSERT_TRUE(fabs(w(0) - v(2)) < 10E-10);
	ASSERT_TRUE(fabs(w(1) - v(3)) < 10E-10);
	ASSERT_TRUE(fabs(w(2) - v(4)) < 10E-10);
	ASSERT_TRUE(fabs(w(3) - v(5)) < 10E-10);
	ASSERT_TRUE(fabs(w(4) - v(6)) < 10E-10);
}


TEST_F(VectorClassTest, mean) {

	rodeo::vec v(3);
	v(0) = 1;
	v(0) = 2;
	v(0) = 3;

	ASSERT_TRUE((v.calculateMean() - 2.0) < 10E-10);
}


TEST_F(VectorClassTest, concatenate) {

	rodeo::vec v1(3);
	v1.fillRandom();
	rodeo::vec v2(2);
	v2.fillRandom();

	// Concatenate vectors v1 and v2
	rodeo::vec v3 = v1.concatenate(v2);
	ASSERT_TRUE(v3.getSize() == 5 );
	ASSERT_TRUE(fabs(v3(3) - v2(0)) < 10E-10);

}

//TEST_F(VectorClassTest, matmul) {
//
//	matrix.resize(2, 3);
//	matrix.fillRandom();
//
//	vector.resize(3);
//	vector.fillRandom();
//	rodeo::vec result = rodeo::vec::matmul(matrix, vector);
//
//	ASSERT_EQ(result.getSize(), matrix.getNRows());
//	ASSERT_NE(result.getSize(), 0);
//
//	ASSERT_EQ(result(0), matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2));
//	ASSERT_EQ(result(1), matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2));
//}
//
//TEST_F(VectorClassTest, vectorMinusMatrixVector) {
//
//	rodeo::mat M(10,10);
//	M.fillRandom();
//	rodeo::vec v(10);
//	v.fillRandom();
//
//	rodeo::vec res = v-rodeo::vec::matmul(M,v);
//
//	res.print();
//
//
//}



