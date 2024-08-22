#include<string>
#include<vector>
#include "../Optimizers/INCLUDE/optimization.hpp"

typedef double (*objectiveFunctionPtrType)(const double *);

struct ParsedConstraintExpression {
    string name;
    string inequality;
    double value;
};

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
	string name;
	string DoEType = "random";
	string cwd;


	void performDoE(void);
	void performDoEForConstraints(void);
	void checkOptimizationSettings();

public:


	unsigned int numberOfConstraints = 0;



	RobustDesignOptimizer();


	void print(void);

	void run(void);
	void setDimension(unsigned int dim);
	void setName(const string nameInput);
	void setCurrentWorkingDirectory(string dir);
	void setDoEStrategy(const std::string& input);
	void setBoxConstraints(double *lb, double *ub);
	void setObjectiveFunction(ObjectiveFunctionPtr, std::string name, std::string filename);
	void addConstraint(ObjectiveFunctionPtr, std::string, std::string);
	void setNameOfTrainingDataFile(std::string name);
	void setDoEOn(unsigned int nSamples);
	void setMaxNumberOfFunctionEvaluations(unsigned int nSamples);



};
