/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), RPTU
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

#include "../INCLUDE/auxiliary_functions.hpp"

#include<gtest/gtest.h>

using namespace std;


TEST(testAuxiliaryFunctions, isNumberBetween){

	double x = 1.2;
	double a = 0.2;
	double b = 4.9;

	ASSERT_TRUE(isNumberBetween(x,a,b));

	int y = 1;
	int c = 2;
	int d = 4;

	ASSERT_FALSE(isNumberBetween(y,c,d));
}

TEST(testAuxiliaryFunctions, isIntheList){

	vector<string> v;
	v.push_back("red");
	v.push_back("green");
	v.push_back("black");


	ASSERT_TRUE(isIntheList<string>(v,"red"));
	ASSERT_FALSE(isIntheList<string>(v,"yellow"));
	ASSERT_FALSE(isIntheList<string>(v,"re"));



	vector<int> w;
	w.push_back(2);
	w.push_back(14);
	w.push_back(77);


	ASSERT_TRUE(isIntheList<int>(w,2));
	ASSERT_TRUE(isNotIntheList<int>(w,44));

}


TEST(testAuxiliaryFunctions, pdf){

	double x = 1.6;
	double sigma = 1.8;
	double mu = 0.9;
	double result = pdf(x, mu, sigma);
	double err =fabs( result-	0.2054931699076307154202);

	EXPECT_LE(err, 10E-10);
}

TEST(testAuxiliaryFunctions, cdf){

	double x = 1.6;
	double sigma = 1.8;
	double mu = 0.9;
	double result = cdf(x, mu, sigma);
	double err =fabs( result -	0.6513208290612620373879);

	EXPECT_LE(err, 10E-10);
}

TEST(testAuxiliaryFunctions, checkifTooCLose){

	rowvec x1(3, fill::randu);

	rowvec x2 = x1*2;
	bool check = checkifTooCLose(x1,x2);
	ASSERT_FALSE(check);

	x2 = x1;
	check = checkifTooCLose(x1,x2);
	ASSERT_TRUE(check);
	x2(0)+= 0.0000001;
	check = checkifTooCLose(x1,x2);
	ASSERT_TRUE(check);

}

TEST(testAuxiliaryFunctions, checkifTooCLoseVectorMatrixVersion){

	rowvec x1(3, fill::randu);
	mat M(20,3,fill::randu);
	M.row(5) = x1;
	bool check = checkifTooCLose(x1,M);
	ASSERT_TRUE(check);

	M.row(5) = 2*x1;
	check = checkifTooCLose(x1,M);
	ASSERT_FALSE(check);

}





