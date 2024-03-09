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
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"


#include<gtest/gtest.h>

using namespace rodeo;

rodeo::vec testFunction(rodeo::vec v){

	return v;
}

class VectorClassTest : public ::testing::Test {
protected:
	void SetUp() override {


	}

	//  void TearDown() override {}


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
	test.print();
}

