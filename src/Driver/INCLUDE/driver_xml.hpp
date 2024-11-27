
#ifndef DRIVER_XML_HPP
#define DRIVER_XML_HPP


#include<vector>
#include<string>
#include "../../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../../Optimizers/INCLUDE/optimization.hpp"
#include "./configkey.hpp"



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

	void setObjectiveFunctionNumberOfTrainingIterations();
	void setBoxConstraints();



public:

	Driver();
	void setConfigFileName(const string &);
	std::vector<std::string> readConstraintFunctionsFromXML(void) const;
	//	std::string readObjectiveFunction(void) const;

	void initializeKeywords(void);
	void addConfigKeysObjectiveFunction(void);
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
