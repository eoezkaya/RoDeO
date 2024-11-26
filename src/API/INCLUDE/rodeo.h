#include<string>
#include<vector>
#include "../../Optimizers/INCLUDE/optimization.hpp"


namespace Rodop{


typedef double (*objectiveFunctionPtrType)(const double *);

struct ParsedConstraintExpression {
    string name;
    string inequality;
    double value;
};

ParsedConstraintExpression parseExpression(const std::string& input);

class RobustDesignOptimizer{

private:
	Optimizer optimizer;
	unsigned int dimension = 0;
	unsigned int numberOfFunctionEvaluations = 0;
	bool startWithDoE = false;
	unsigned int numberOfSamplesForDoE = 0;

	string objectiveFunctionTrainingFilename;
	vector<string> constraintsTrainingDataFilename;

	bool isDimensionSpecified = false;
	bool isObjectiveFunctionSpecified = false;
	bool areBoxConstraintsSpecified = false;

	vector<double> lowerBounds;
	vector<double> upperBounds;

	objectiveFunctionPtrType objectiveFunctionPtr;

	vector<objectiveFunctionPtrType> constraintFunctionPtr;

	mat samplesInput;

	string cwd;
	string DoEType = "random";
	string name;


	void performDoE(void);
	void performDoEForConstraints(void);
	void checkOptimizationSettings();
	void generateDoESamplesInput();

public:



	unsigned int numberOfConstraints = 0;



	RobustDesignOptimizer();


	void print(void);

	void run(void);
	void setDimension(unsigned int dim);
	void setCurrentWorkingDirectory(string dir);
	void setDoEStrategy(const std::string& input);
	void setName(const std::string& nameInput);
	void setBoxConstraints(double *lb, double *ub);
	void setObjectiveFunction(ObjectiveFunctionPtr, std::string name, std::string filename);
	void addConstraint(ObjectiveFunctionPtr, std::string, std::string);
	void setNameOfTrainingDataFile(std::string name);
	void setDoEOn(unsigned int nSamples);
	void setMaxNumberOfFunctionEvaluations(unsigned int nSamples);



};


} /* Namespace Rodop */

