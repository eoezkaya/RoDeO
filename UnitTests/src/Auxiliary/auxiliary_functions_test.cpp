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

#include "auxiliary_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef TEST_AUX



TEST(testAuxiliaryFunctions, isBetween){

	double x = 1.2;
	double a = 0.2;
	double b = 4.9;

	ASSERT_TRUE(isBetween(x,a,b));

	int y = 1;
	int c = 2;
	int d = 4;

	ASSERT_FALSE(isBetween(y,c,d));
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

TEST(testAuxiliaryFunctions, isIntheListV2){

	int *v = new int[4];
	v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 7;


	ASSERT_TRUE(isIntheList<int>(v,2,4));
	ASSERT_TRUE(isNotIntheList<int>(v,44,4));

	delete[] v;

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


TEST(testAuxiliaryFunctions, removeSpacesFromString){

	std::string testString = " this is a test string ";

	testString = removeSpacesFromString(testString);

	int ifEqual = testString.compare("thisisateststring");
	ASSERT_EQ(ifEqual,0);


}


TEST(testAuxiliaryFunctions, getStringValuesFromString){

	std::string testString = "apple";

	std::vector<std::string> values;

	values = getStringValuesFromString(testString, ',');


	ASSERT_EQ(values[0],"apple");

	testString = "apple, 1.22, banana";
	values = getStringValuesFromString(testString, ',');
	ASSERT_EQ(values[2],"banana");



}


TEST(testAuxiliaryFunctions, getDoubleValuesFromString){

	std::string testString = "-1.99";

	vec values;

	values = getDoubleValuesFromString(testString, ',');


	ASSERT_EQ(values(0),-1.99);

	testString = "-1,   1.22, 0.001";
	values = getDoubleValuesFromString(testString, ',');
	ASSERT_EQ(values(2),0.001);



}

TEST(testAuxiliaryFunctions, removeKeywordFromString){

	std::string key = "DIMENSION";
	std::string s = "DIMENSION = blabla , blabla";

	std::string  s1 = removeKeywordFromString(s, key);

	int ifEqual = s1.compare("blabla,blabla");
	ASSERT_EQ(ifEqual,0);

	key = "NOTAKEYWORD";

	std::string  s2 = removeKeywordFromString(s, key);
	ifEqual = s2.compare(s);
	ASSERT_EQ(ifEqual,0);


}

TEST(testAuxiliaryFunctions, isEmpty){

	std::string testStr;

	bool ifEmpty = isEmpty(testStr);

	ASSERT_EQ(ifEmpty , true);

}


TEST(testAuxiliaryFunctions, isNotEmpty){

	std::string testStr = "test";

	bool ifNotEmpty = isNotEmpty(testStr);

	ASSERT_EQ(ifNotEmpty , true);


}


TEST(testAuxiliaryFunctions, isEqualString){

	std::string testStr = "test";
	std::string str = "test";
	std::string str2 = "tes";
	std::string str3 = "tesa";

	ASSERT_TRUE(isEqual(testStr,str));
	ASSERT_FALSE(isEqual(testStr,str2));
	ASSERT_FALSE(isEqual(testStr,str3));

}

#endif

