#include<string>
#include<vector>
#include <memory>

typedef double (*objectiveFunctionPtrType)(const double *);
class RobustDesignOptimizer;
class RobustDesignOptimizerAPI{
public:
	RobustDesignOptimizerAPI();
	~RobustDesignOptimizerAPI();
	void print();

	void run();
	void setDimension(unsigned int dim);
	void setName(const std::string nameInput);
	void setCurrentWorkingDirectory(std::string dir);
	void setDoEStrategy(const std::string& input);
	void setBoxConstraints(double *lb, double *ub);
	void setObjectiveFunction(objectiveFunctionPtrType, std::string name, std::string filename);
	void addConstraint(objectiveFunctionPtrType, std::string, std::string);
	void setNameOfTrainingDataFile(std::string name);
	void setDoEOn(unsigned int nSamples);
	void setMaxNumberOfFunctionEvaluations(unsigned int nSamples);
private:
	std::unique_ptr<RobustDesignOptimizer> impl;
};
