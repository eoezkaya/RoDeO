#ifndef GP_OPTIMIZER_HPP
#define GP_OPTIMIZER_HPP

#include<string>
#include<vector>
#include "../../LinearAlgebra/INCLUDE/vector.hpp"


namespace Rodop{


typedef double (*GeneralPurposeObjectiveFunction)(double *);

class GeneralPurposeOptimizer{


protected:

	unsigned int dimension = 0;

	vec lowerBounds;
	vec upperBounds;
	std::string problemName;
	std::string filenameOptimizationHistory;
	std::string filenameWarmStart;
	std::string filenameOptimizationResult;

	unsigned int maxNumberOfFunctionEvaluations = 0.0;

	double (*calculateObjectiveFunction)(double *);

	bool ifObjectiveFunctionIsSet = false;
	bool ifProblemNameIsSet = false;
	bool ifFilenameOptimizationHistoryIsSet = false;
	bool ifFilenameWarmStartIsSet = false;
	bool ifFilenameOptimizationResultIsSet = false;
	bool ifWarmStart = false;
	bool ifSaveWarmStartFile = false;
	bool ifBoundsAreSet = false;
	bool ifScreenDisplay = false;

	void checkName(std::string name) const;

public:

	void setDimension(unsigned int dim);
	unsigned int getDimension(void) const;


	void setBounds(double, double);
	void setBounds(std::vector<double>, std::vector<double>);
	void setBounds(const vec &lb, const vec &ub);

	bool areBoundsSet(void) const;

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setProblemName(std::string);
	void setFilenameOptimizationHistory(std::string);
	void setFilenameWarmStart(std::string);
	void setFilenameOptimizationResult(std::string);


	bool isProblemNameSet(void);
	bool isFilenameOptimizationHistorySet(void);
	bool isFilenameWarmStartSet(void);
	bool isFilenameOptimizationResultSet(void);

	void setObjectiveFunction(GeneralPurposeObjectiveFunction );
	bool isObjectiveFunctionSet(void) const;

	void setMaxNumberOfFunctionEvaluations(unsigned int);

	virtual void optimize(void);
	virtual double calculateObjectiveFunctionInternal(const vec &);
	virtual void writeWarmRestartFile(void);

	double callObjectiveFunction(const vec &);


};

} /* Namespace Rodop */

#endif
