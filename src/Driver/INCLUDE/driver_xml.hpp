
#ifndef DRIVER_XML_HPP
#define DRIVER_XML_HPP


#include<vector>
#include<string>
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../../Optimizers/INCLUDE/optimization.hpp"
#include "../../SurrogateModels/INCLUDE/surrogate_model_tester.hpp"
#include "./configkey.hpp"

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif


namespace Rodop{

class Parameter{

public:
	std::string name;
	std::string type;
	double lb = 0.0;
	double ub = 1.0;
	double increment = 0.0;

	void print(void){
		std::cout<<"name = " <<name <<"\n";
		std::cout<<"type = " <<type <<"\n";
		std::cout<<"lb = " <<lb <<"\n";
		std::cout<<"ub = " <<ub <<"\n";
		std::cout<<"increment = " <<increment <<"\n";
	}


};


class Driver{


#ifdef UNIT_TESTS
	friend class DriverXMLTest;
	FRIEND_TEST(DriverXMLTest, constructor);
	FRIEND_TEST(DriverXMLTest, setConfigFileName);
	FRIEND_TEST(DriverXMLTest, readConstraintFunctions);
	FRIEND_TEST(DriverXMLTest, readObjectiveFunction);
	FRIEND_TEST(DriverXMLTest, addConfigKeysObjectiveFunction);

	FRIEND_TEST(DriverXMLTest, getConfigKeyValueString);
	FRIEND_TEST(DriverXMLTest, getConfigKeyValueStringNotFound);
	FRIEND_TEST(DriverXMLTest, readObjectiveFunctionKeywordsFromXML);
	FRIEND_TEST(DriverXMLTest, readGeneralSettings);
	FRIEND_TEST(DriverXMLTest, readOptimizationParameters);
	FRIEND_TEST(DriverXMLTest, readOptimizationParametersThrowsExceptionInvalidLowerBound);
	FRIEND_TEST(DriverXMLTest, readOptimizationParametersThrowsExceptionInvalidIncrement);


	FRIEND_TEST(DriverXMLTest, readObjectiveFunctionKeywords);
	FRIEND_TEST(DriverXMLTest, readConfigurationFileWithConstraint);

	FRIEND_TEST(DriverXMLTest, getConfigKeyValueIntReturnsCorrectValueWhenKeyExists);
	FRIEND_TEST(DriverXMLTest, getConfigKeyValueIntThrowsOnEmptyKey);
	FRIEND_TEST(DriverXMLTest, getConfigKeyValueIntReturnsMinIntWhenKeyDoesNotExist);


	FRIEND_TEST(DriverXMLTest,getConfigKeyValueDoubleThrowsOnEmptyList);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueDoubleReturnsCorrectValueWhenKeyExists);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueDoubleReturnsDefaultValueWhenKeyDoesNotExist);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueDoubleHandlesVariousFloatingPointValues);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueDoubleThrowsExceptionOnEmptyKey);

	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringThrowsOnEmptyList);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringReturnsCorrectValueWhenKeyExists);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringReturnsEmptyStringWhenKeyDoesNotExist);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringKeyCaseSensitivity);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringThrowsExceptionOnEmptyKey);

	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringVectorThrowsVectorOnEmptyList);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringVectorReturnsCorrectVectorWhenKeyExists);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringVectorReturnsEmptyVectorWhenKeyDoesNotExist);
	FRIEND_TEST(DriverXMLTest,getConfigKeyValueStringVectorThrowsExceptionOnEmptyKey);



#endif


private:

	Optimizer optimizationStudy;
	ObjectiveFunction objectiveFunction;

	bool isConfigFileSet = false;
	bool areConfigKeysInitialized = false;

	bool isOptimization = false;
	bool isSurrogateModelTest = false;

	bool isConfigurationFileRead = false;

	string configFilename;

	vector<string> availableSurrogateModels;
	vector<Keyword> keywordsObjectiveFunction;
	vector<Keyword> keywordsConstraintFunction;
	vector<Keyword> keywordsGeneral;

	vector<Parameter> designParameters;

	Bounds boxConstraints;


	void addConfigKey(vector<Keyword> &list, string name, string type);
	void readObjectiveFunctionKeywords(void);
	void readConstraintFunctionKeywords(string inputText);

	std::string getConfigKeyValueString(vector<Keyword> &list, const std::string &xml_input, const std::string &key);


	int getConfigKeyValueInt(vector<Keyword>& list, const string& key);
	double getConfigKeyValueDouble(vector<Keyword> &list, const string& key);
	string getConfigKeyValueString(vector<Keyword> &list, const string& key);

	vector<string> getConfigKeyValueStringVector(vector<Keyword> &list, const string& key);
	vector<double> getConfigKeyValueDoubleVector(vector<Keyword> &list, const string& key);

	void readGeneralSettings(void);
	void setDimensionOptimizationStudy(void);
	void setNumberOfFunctionEvaluationsOptimizationStudy(void);
	void setMaxNumberOfInnerIterationsOptimizationStudy(void);
	void setNameOptimizationStudy();
	void setParameterNamesOptimizationStudy();
	void setBoxConstraintsOptimizationStudy();
	void setDiscreteParametersOptimizationStudy();
	void setNumberOfThreadsOptimizationStudy(void);

	void readOptimizationParameters(void);

	void readObjectiveFunction();
	void setObjectiveFunctionDefinition(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionDefinitionMultiFidelity(
			ObjectiveFunctionDefinition &definition);

	void readConstraintFunctions(void);


	void setConstraintName(ConstraintDefinition &constraintDefinition);
	void setConstraintType(ConstraintDefinition &constraintDefinition);
	void setConstraintValue(ConstraintDefinition &constraintDefinition);
	void setConstraintUDF(ConstraintFunction &constraint);
	void setConstraintNumberOfTrainingIterations(ConstraintFunction &constraint);

	void setConstraintDesignVectorFilename(ObjectiveFunctionDefinition &definition);
	void setConstraintExecutableFilename(ObjectiveFunctionDefinition &definition);
	void setConstraintOutputFilename(ObjectiveFunctionDefinition &definition);
	void setConstrantFunctionTrainingDataFilename(ObjectiveFunctionDefinition &definition);

	void setConstraintSurrogateModelType(ObjectiveFunctionDefinition &definition);
	void setConstraintFunctionGradientExecutableName(
			ObjectiveFunctionDefinition &definition);

	void setConstraintFunctionGradientOutputFilename(
			ObjectiveFunctionDefinition &definition);

	void setConstraintFunctionExecutableFilenamesMF(
			ObjectiveFunctionDefinition &definition);
	void setConstraintFunctionTrainingDataFilenamesMF(
			ObjectiveFunctionDefinition &definition);
	void setConstraintFunctionOutputFilenamesMF(
			ObjectiveFunctionDefinition &definition);
	void setConstraintFunctionSurrogateTypesMF(
			ObjectiveFunctionDefinition &definition);




	void setObjectiveFunctionName(ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionDesignVectorFilename(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionExecutableName(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionTrainingDataFilename(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionOutputFilename(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionHiFiSurrogateModelType(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionGradientExecutableName(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionGradientOutputFilename(
			ObjectiveFunctionDefinition &definition);
	void readConstraintFunctionsSingleFidelity(
			ObjectiveFunctionDefinition &definition);
	void readConstraintFunctionsMultiFidelity(
			ObjectiveFunctionDefinition &definition);
	void setObjectiveFunctionNumberOfTrainingIterations();
	void setBoxConstraints();



public:

	Driver();
	void setConfigFileName(const string &);
	std::vector<std::string> readConstraintFunctionsFromXML(void) const;
	//	std::string readObjectiveFunction(void) const;

	void initializeKeywords(void);
	void addConfigKeysObjectiveFunctionMultiFidelity(void);
	void addConfigKeysObjectiveFunction(void);

	void addConfigKeysConstraintFunctionMultiFidelity(void);
	void addConfigKeysConstraintFunction(void);

	void addConfigKeysGeneralSettings(void);


	void readConfigurationFile(void);

	SURROGATE_MODEL getSurrogateModelID(const std::string& modelName) const;

	std::string readASegmentFromXMLFile(string keyword) const;

	void runOptimization(void);

	void run(void);

};



} /* Namespace Rodop */

#endif
