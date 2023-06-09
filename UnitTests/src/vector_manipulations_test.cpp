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
#include "vector_manipulations.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>


#ifdef VECTOR_MANIP_TEST

TEST(testVectorManipulations, makeUnitVector){

	vec v(10,fill::randu);
	v = v*5;
	vec w = makeUnitVector(v);
	double normw = norm(w,2);
	EXPECT_LT(fabs(normw-1.0),10E-10);
}

TEST(testVectorManipulations, makeUnitRowVector){

	rowvec v(10,fill::randu);
	v = v*5;
	rowvec w = makeUnitVector(v);

	double normw = norm(w,2);
	EXPECT_LT(fabs(normw-1.0),10E-10);

}

TEST(testVectorManipulations, normalizeVector){

	rowvec x(5,fill::randu);
	vec xmin(5); xmin.fill(0.1);
	vec xmax(5); xmax.fill(2.1);

	rowvec xNormalized = normalizeVector(x, xmin, xmax);
	double xCheck = xNormalized(0)*5 * (2.0) + 0.1;

	double error = fabs(xCheck-x(0));
	EXPECT_LT(error, 10E-10);

	x(0) = 1.3; x(1) = 10.7; x(2) = -1.3; x(3) = 0.0; x(4) = 1.7;
	xmin.fill(1.3);
	xmax.fill(50.0);
	xNormalized = normalizeVector(x, xmin, xmax);
	EXPECT_LT(fabs(xNormalized(0)), 10E-10);

}

TEST(testVectorManipulations, normalizeRowVectorBack){

	rowvec x(5,fill::randu);
	vec xmin(5); xmin.fill(0.1);
	vec xmax(5); xmax.fill(2.1);

	rowvec xNormalized = normalizeVector(x, xmin, xmax);
	rowvec xNormalizedBack = normalizeVectorBack(xNormalized, xmin, xmax);

	for(unsigned int i=0; i<5; i++){
		double err = fabs(x(i) - xNormalizedBack(i));
		EXPECT_LT(err,10E-10);

	}



}

TEST(testVectorManipulations, addOneElement){

	vec testVector(10,fill::randu);
	double val = testVector(5);

	double someValue = 1.987;
	addOneElement<vec>(testVector, someValue);
	ASSERT_TRUE(testVector.size() == 11);
	double error  = fabs(testVector(5) - val);
	ASSERT_LT(error, 10E-10);
	error  = fabs(testVector(10) - someValue);
	ASSERT_LT(error, 10E-10);

	vec testRowVector(10,fill::randu);
	val = testRowVector(5);
	addOneElement<vec>(testRowVector, someValue);
	ASSERT_TRUE(testRowVector.size() == 11);

	error  = fabs(testRowVector(5) - val);
	ASSERT_LT(error, 10E-10);
	error  = fabs(testRowVector(10) - someValue);
	ASSERT_LT(error, 10E-10);


}

TEST(testVectorManipulations, copyVector){

	vec a(10,fill::randu);
	vec b(7,fill::randu);

	copyVector(a,b);

	for(unsigned int i=0; i<7; i++){

		double error  = fabs(a(i) - b(i));
		ASSERT_LT(error, 10E-10);


	}

}


TEST(testVectorManipulations, copyRowVector){

	rowvec a(10,fill::randu);
	rowvec b(7,fill::randu);

	copyVector(a,b);

	for(unsigned int i=0; i<7; i++){
		double error  = fabs(a(i) - b(i));
		ASSERT_LT(error, 10E-10);
	}

}


TEST(testVectorManipulations, copyVectorAfterIndex){

	rowvec a(5,fill::randu);
	rowvec b(2,fill::randu);

	copyVector(a,b,3);


	for(unsigned int i=3; i<5; i++){
		double error  = fabs(a(i) - b(i-3));
		ASSERT_LT(error, 10E-10);
	}

}





#endif
