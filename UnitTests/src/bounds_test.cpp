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

#include<gtest/gtest.h>
#include <armadillo>
#include "bounds.hpp"

using namespace arma;

#define TEST_BOUNDS
#ifdef TEST_BOUNDS

TEST(testBounds, testBoundsConstructor){

	unsigned int dim = 5;
	Bounds testBounds(dim);

	unsigned int dimGet = testBounds.getDimension();
	ASSERT_EQ(dimGet,dim);


}

TEST(testBounds, testBoundsConstructorWithVectors){

	unsigned int dim = 5;
	vec lowerBounds = zeros<vec>(dim);
	vec upperBounds = ones<vec>(dim);

	Bounds testBounds(lowerBounds,upperBounds);

	vec lowerBoundsGet = testBounds.getLowerBounds();

	for(unsigned int i=0; i<dim; i++) ASSERT_EQ(lowerBoundsGet(i),lowerBounds(i));

	vec upperBoundsGet = testBounds.getUpperBounds();
	for(unsigned int i=0; i<dim; i++) ASSERT_EQ(upperBoundsGet(i),upperBounds(i));

}


TEST(testBounds, testsetBoundsWithVectors){

	unsigned int dim = 5;
	vec lowerBounds = zeros<vec>(dim);
	vec upperBounds = ones<vec>(dim);

	Bounds testBounds(dim);

	testBounds.setBounds(lowerBounds,upperBounds);

	vec lowerBoundsGet = testBounds.getLowerBounds();
	for(unsigned int i=0; i<dim; i++) ASSERT_EQ(lowerBoundsGet(i),lowerBounds(i));



}

TEST(testBounds, testsetBoundsWithDoubles){

	unsigned int dim = 5;
	Bounds testBounds(dim);
	double lowerBound = 0.0;
	double upperBound = 1.0;
	testBounds.setBounds(lowerBound,upperBound);

}


TEST(testBounds, testisBoundsAreSet){

	unsigned int dim = 5;
	Bounds testBounds(dim);
	double lowerBound = 0.0;
	double upperBound = 1.0;
	testBounds.setBounds(lowerBound,upperBound);

	bool result = testBounds.areBoundsSet();
	ASSERT_EQ(result, true);

}


TEST(testBounds, testcheckIfBoundsAreValid){

	unsigned int dim = 5;
	vec lowerBound = zeros<vec>(dim);
	vec upperBound = ones<vec>(dim);

	Bounds testBounds(lowerBound,upperBound);

	bool result = testBounds.checkIfBoundsAreValid();
	ASSERT_EQ(result, true);

}

TEST(testBounds, testlowerBound){

	unsigned int dim = 5;
	Bounds testBounds(dim);
	vec lowerBounds = zeros<vec>(dim);
	vec upperBounds = ones<vec>(dim);
	lowerBounds(2) = 0.33;

	testBounds.setBounds(lowerBounds,upperBounds);

	double result1 = testBounds.getLowerBound(2);
	EXPECT_EQ(result1,lowerBounds(2));

	double result2 = testBounds.getUpperBound(4);
	EXPECT_EQ(result2,upperBounds(4));


}

TEST(testBounds, testisPointWithinBounds){

	unsigned int dim = 5;
	vec lowerBound = zeros<vec>(dim);
	vec upperBound = ones<vec>(dim);

	Bounds testBounds(lowerBound,upperBound);

	vec point(dim, fill::randu);

	bool result = testBounds.isPointWithinBounds(point);
	ASSERT_EQ(result,true);



}




#endif
