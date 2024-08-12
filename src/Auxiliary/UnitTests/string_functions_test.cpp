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

#include "auxiliary_functions.hpp"
#include<gtest/gtest.h>


TEST(testStringFunctions, checkIfOn){

	std::string testString = "Yes";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "Yeah";
	ASSERT_FALSE(checkIfOn(testString));
	testString = "Y";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "YES";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "ON";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "on";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "On";
	ASSERT_TRUE(checkIfOn(testString));
	testString = "yes";
	ASSERT_TRUE(checkIfOn(testString));


}

TEST(testStringFunctions, checkIfOff){

	std::string testString = "No";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "Nope";
	ASSERT_FALSE(checkIfOff(testString));
	testString = "N";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "No";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "OFF";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "off";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "Off";
	ASSERT_TRUE(checkIfOff(testString));
	testString = "no";
	ASSERT_TRUE(checkIfOff(testString));


}



TEST(testStringFunctions, removeSpacesFromString){

	std::string testString = " this is a test string ";

	testString = removeSpacesFromString(testString);

	int ifEqual = testString.compare("thisisateststring");
	ASSERT_EQ(ifEqual,0);


}


TEST(testStringFunctions, getStringValuesFromString){

	std::string testString = "apple";

	std::vector<std::string> values;

	values = getStringValuesFromString(testString, ',');


	ASSERT_EQ(values[0],"apple");

	testString = "apple, 1.22, banana";
	values = getStringValuesFromString(testString, ',');
	ASSERT_EQ(values[2],"banana");



}


TEST(testStringFunctions, getDoubleValuesFromString){

	std::string testString = "-1.99";

	vec values;

	values = getDoubleValuesFromString(testString, ',');


	ASSERT_EQ(values(0),-1.99);

	testString = "-1,   1.22, 0.001";
	values = getDoubleValuesFromString(testString, ',');
	ASSERT_EQ(values(2),0.001);



}

TEST(testStringFunctions, removeKeywordFromString){

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

TEST(testStringFunctions, isEmpty){

	std::string testStr;

	bool ifEmpty = isEmpty(testStr);

	ASSERT_EQ(ifEmpty , true);

}


TEST(testStringFunctions, isNotEmpty){

	std::string testStr = "test";

	bool ifNotEmpty = isNotEmpty(testStr);

	ASSERT_EQ(ifNotEmpty , true);


}


TEST(testStringFunctions, isEqualString){

	std::string testStr = "test";
	std::string str = "test";
	std::string str2 = "tes";
	std::string str3 = "tesa";

	ASSERT_TRUE(isEqual(testStr,str));
	ASSERT_FALSE(isEqual(testStr,str2));
	ASSERT_FALSE(isEqual(testStr,str3));

}

TEST(testStringFunctions, convertToString){

	double someNumber = 0.0000123124;
	std::string sameNumber = convertToString(someNumber,12);
	ASSERT_TRUE(sameNumber == "0.000012312400");
}


