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

#include<gtest/gtest.h>
#include "../INCLUDE/random_functions.hpp"


TEST(testrandom, generateRandomDouble){

	double r = generateRandomDouble(12.7, 12.8);
	ASSERT_TRUE(r>12.7);
	ASSERT_TRUE(r<12.8);


}

TEST(testrandom, generateRandomvector){

	vec lb(3);
	vec ub(3);

	lb.fill(-1.0);
	ub.fill(2.8);

	vec r = generateRandomVector<vec>(lb,ub);

	ASSERT_TRUE(r.size() == 3);

	for(unsigned int i=0; i<3; i++){

		ASSERT_TRUE(r(i) > -1.0);
		ASSERT_TRUE(r(i) <  2.8);

	}


}

TEST(testrandom, generateRandomRowvector){

	vec lb(3);
	vec ub(3);

	lb.fill(-1.0);
	ub.fill(2.8);

	rowvec r = generateRandomVector<rowvec>(lb,ub);

	ASSERT_TRUE(r.size() == 3);

	for(unsigned int i=0; i<3; i++){

		ASSERT_TRUE(r(i) > -1.0);
		ASSERT_TRUE(r(i) <  2.8);

	}


	r = generateRandomVector<rowvec>(-1.0,2.8,3);

	ASSERT_TRUE(r.size() == 3);

	for(unsigned int i=0; i<3; i++){

		ASSERT_TRUE(r(i) > -1.0);
		ASSERT_TRUE(r(i) <  2.8);

	}



}



