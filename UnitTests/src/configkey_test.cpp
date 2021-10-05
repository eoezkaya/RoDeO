/*
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


#include "configkey.hpp"
#include<gtest/gtest.h>



TEST(testConfigkey, testConfigKeysetValue){

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	EXPECT_EQ(testKey.doubleValue, 5.2);


	ConfigKey testKey2("TESTKEY2","int");
	number = "14";
	testKey2.setValue(number);
	EXPECT_EQ(testKey2.intValue, 14);



}

TEST(testConfigkey, testConfigKeyListadd){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");

	testList.add(testKey);

	EXPECT_EQ(testList.countNumberOfElements(), 1);



}

TEST(testConfigkey, testConfigKeyListgetConfigKey){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey returnedKey = testList.getConfigKey(0);

	EXPECT_EQ(returnedKey.doubleValue, 5.2);


}

TEST(testConfigkey, testConfigKeyListgetConfigKey2){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","string");
	std::string value("something");
	testKey2.setValue(value);

	testList.add(testKey2);


	ConfigKey returnedKey = testList.getConfigKey("TESTKEY");

	EXPECT_EQ(returnedKey.doubleValue, 5.2);

}

TEST(testConfigkey, testConfigKeyListifKeyIsAlreadyIntheList){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY","string");
	std::string value("something");

	bool ifIsIntheList = testList.ifKeyIsAlreadyIntheList(testKey2);

	EXPECT_EQ(ifIsIntheList, true);

	ConfigKey testKey3("TESTKEY3","string");

	ifIsIntheList = testList.ifKeyIsAlreadyIntheList(testKey3);

	EXPECT_EQ(ifIsIntheList, false);


}




TEST(testConfigkey, testConfigKeyListgetConfigKeyStringValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","string");
	std::string value("something");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	std::string keyValue = testList.getConfigKeyStringValue("TESTKEY3");

	EXPECT_TRUE(keyValue.empty());

	keyValue = testList.getConfigKeyStringValue("TESTKEY2");
	EXPECT_EQ(keyValue,"something");



}

TEST(testConfigkey, testConfigKeyListgetConfigKeyIntValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","int");
	std::string value("2");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	int keyValue = testList.getConfigKeyIntValue("TESTKEY2");

	EXPECT_EQ(keyValue,2);


}


TEST(testConfigkey, testConfigKeyListgetConfigKeyDoubleValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","int");
	std::string value("2");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	double keyValue = testList.getConfigKeyDoubleValue("TESTKEY");

	EXPECT_EQ(keyValue,5.2);


}


TEST(testConfigkey, testConfigKeyListgetConfigKeyVectorStringValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","stringVector");
	std::string value("{val1, val2, val3");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	std::vector<std::string> valueVector = testList.getConfigKeyVectorStringValue("TESTKEY2");

	EXPECT_EQ(valueVector[0],"val1");
	EXPECT_EQ(valueVector[1],"val2");
	EXPECT_EQ(valueVector[2],"val3");


}

TEST(testConfigkey, testConfigKeyListgetConfigKeyVectorDoubleValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","doubleVector");
	std::string value("{2.2, 1.1, -0.1");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	vec valueVector = testList.getConfigKeyVectorDoubleValue("TESTKEY2");

	EXPECT_EQ(valueVector[0],2.2);
	EXPECT_EQ(valueVector[1],1.1);
	EXPECT_EQ(valueVector[2],-0.1);

}


TEST(testConfigkey, testConfigKeyListgetConfigKeyStringVectorValueAtIndex){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	std::string number("5.2");
	testKey.setValue(number);

	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","stringVector");
	std::string value("{val1, val2, val3");
	testKey2.setValue(value);

	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	value = testList.getConfigKeyStringVectorValueAtIndex("TESTKEY2",1);

	EXPECT_EQ(value,"val2");


}

TEST(testConfigkey, testConfigKeyListassignKeywordValue){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","stringVector");
	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	std::pair<std::string, std::string> input = std::make_pair("TESTKEY2", "{val1, val2, val3}");

	testList.assignKeywordValue(input);

	std::string value = testList.getConfigKeyStringVectorValueAtIndex("TESTKEY2",1);

	EXPECT_EQ(value,"val2");


}

TEST(testConfigkey, testConfigKeyListassignKeywordValue2){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","stringVector");
	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	testList.assignKeywordValue(std::string("TESTKEY2"),std::string("{val1, val2, val3}"));

	std::string value = testList.getConfigKeyStringVectorValueAtIndex("TESTKEY2",1);

	EXPECT_EQ(value,"val2");


}

TEST(testConfigkey, testConfigKeyListassignKeywordValue3){

	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","doubleVector");
	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	vec testVec(3);
	testVec(0) = 1.2;
	testVec(1) = 2.2;
	testVec(2) = 3.2;

	testList.assignKeywordValue("TESTKEY2",testVec);

	vec values = testList.getConfigKeyVectorDoubleValue("TESTKEY2");

	EXPECT_EQ(values(1),2.2);

}

TEST(testConfigkey, testassignKeywordValueWithIndex){

	ConfigKeyList testList;
	ConfigKey testKey1("PROBLEM_TYPE","string");
	testList.add(testKey1);
	ConfigKey testKey2("DIMENSION","int");
	testList.add(testKey2);

	std::string type = "MINIMIZATION";
	testList.assignKeywordValueWithIndex(type,0);

	ConfigKey testKey = testList.getConfigKey(0);
	int ifEqual = testKey.stringValue.compare("MINIMIZATION");
	ASSERT_EQ(ifEqual,0);

	testList.assignKeywordValueWithIndex("5",1);
	testKey = testList.getConfigKey("DIMENSION");

	ASSERT_EQ(testKey.intValue,5);


}



TEST(testConfigkey, testsearchConfigKeywordInString){


	ConfigKeyList testList;

	ConfigKey testKey("TESTKEY","double");
	testList.add(testKey);

	ConfigKey testKey2("TESTKEY2","doubleVector");
	testList.add(testKey2);

	ConfigKey testKey3("TESTKEY3","string");
	testList.add(testKey3);

	std::string str = "TESTKEY 3";
	int foundIndex = testList.searchKeywordInString(str);
	EXPECT_EQ(foundIndex, -1);
	str = "TESTKEY=3";
	foundIndex = testList.searchKeywordInString(str);
	EXPECT_EQ(foundIndex, 0);


}
TEST(testConfigkey, testifFeatureIsOn){

	ConfigKeyList testList;

	ConfigKey testKey("WARM_START","string");
	testList.add(testKey);
	std::string s1 = "WARM_START";
	std::string s2 = "N";
	testList.assignKeywordValue(s1,s2);
	bool ifWarmStart = testList.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, false);

	s2 = "Yes";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "YES";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "Yeah";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOn("WARM_START");
	EXPECT_EQ(ifWarmStart, false);


}


TEST(testConfigkey, testifFeatureIsOff){

	ConfigKeyList testList;

	ConfigKey testKey("WARM_START","string");
	testList.add(testKey);
	std::string s1 = "WARM_START";
	std::string s2 = "N";
	testList.assignKeywordValue(s1,s2);
	bool ifWarmStart = testList.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "Yes";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, false);

	s2 = "NO";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

	s2 = "no";
	testList.assignKeywordValue(s1,s2);
	ifWarmStart = testList.ifFeatureIsOff("WARM_START");
	EXPECT_EQ(ifWarmStart, true);

}

TEST(testConfigkey, testparseString){

	ConfigKeyList testList;
	testList.add(ConfigKey("NAME","string") );
	testList.add(ConfigKey("DESIGN_VECTOR_FILE","string") );

	testList.add(ConfigKey("OUTPUT_FILE","stringVector") );
	testList.add(ConfigKey("PATH","stringVector") );
	testList.add(ConfigKey("GRADIENT","stringVector") );
	testList.add(ConfigKey("EXECUTABLE","stringVector") );
	testList.add(ConfigKey("MARKER","stringVector") );
	testList.add(ConfigKey("MARKER_FOR_GRADIENT","stringVector") );
	testList.add(ConfigKey("NUMBER_OF_TRAINING_ITERATIONS","int") );
	testList.add(ConfigKey("MULTILEVEL_SURROGATE","string") );

	std::string testStr="\nNAME = HimmelblauObjectiveFunction\nEXECUTABLE = himmelblau\nOUTPUT_FILE = objFunVal.dat\nDESIGN_VECTOR_FILE = dv.dat\nMARKER = J";


	testList.parseString(testStr);

	std::string name = testList.getConfigKeyStringValue("NAME");
	EXPECT_EQ(name, "HimmelblauObjectiveFunction");

	std::string exe = testList.getConfigKeyStringVectorValueAtIndex("EXECUTABLE",0);
	EXPECT_EQ(exe, "himmelblau");

}
