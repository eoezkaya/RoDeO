#ifndef CONFIGKEY_HPP
#define CONFIGKEY_HPP

#include<vector>
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../../Optimizers/INCLUDE/optimization.hpp"
#include "../../INCLUDE/globals.hpp"
namespace Rodop{

class Keyword{

#ifdef UNIT_TESTS

	friend class KeywordTest;
	friend class DriverXMLTest;
	FRIEND_TEST(KeywordTest, constructor);
	FRIEND_TEST(KeywordTest, getValueFromXMLStringDouble);
	FRIEND_TEST(KeywordTest, getValueFromXMLStringString);
	FRIEND_TEST(KeywordTest, getValueFromXMLStringInt);


	FRIEND_TEST(DriverXMLTest,setSomeKeywords);


#endif

public:

    Keyword() {}
	Keyword(const std::string& n, const std::string& t, double v) : name(n), type(t), valueDouble(v) {}
	Keyword(const std::string& n, const std::string& t, int v) : name(n), type(t), valueInt(v) {}
	Keyword(const std::string& n, const std::string& t, const std::string& v) : name(n), type(t), valueString(v) {}
	Keyword(const std::string& n, const std::string& t, const vector<std::string>& v) : name(n), type(t), valueStringVector(v) {}




	void returnErrorIfNotSet(void) const;
	void setType(string whichType);
	void setName(string name);
	string getName(void) const;

	void getValueFromXMLString(const string &input);
	void print(void) const;
	void printToLog() const;

	string getStringValue(void) const;
	vector<string> getStringValueVector(void) const;

	double getDoubleValue(void) const;
	int getIntValue(void) const;

	vector<double> getDoubleValueVector(void) const;

	string getStringValueVectorAt(unsigned indx) const;
	double getDoubleValueVectorAt(unsigned indx) const;

private:


	string name = "";
	string type = "";
	double valueDouble = EPSILON;
	string valueString = "";
	vector<string> valueStringVector;
	vector<double> valueDoubleVector;
	int valueInt = NONEXISTINGINTKEYWORD;
	vector<int> valueIntVector;

	bool isValueSet = false;


};




} /* Namespace Rodop */

#endif
