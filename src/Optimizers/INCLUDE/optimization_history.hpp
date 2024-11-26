#ifndef OPT_HISTORY_HPP
#define OPT_HISTORY_HPP

#include "../../Design/INCLUDE/design.hpp"

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif

namespace Rodop{

class OptimizationHistory{

#ifdef UNIT_TESTS
	friend class OptimizationHistoryTest;
	FRIEND_TEST(OptimizationHistoryTest, constructor);
	FRIEND_TEST(OptimizationHistoryTest, setHeader);
	FRIEND_TEST(OptimizationHistoryTest, calculateCrowdingFactor);
#endif

private:

	string filename = "optimizationHistory.csv";
	string objectiveFunctionName = "objective";
	vector<string> constraintNames;
	vector<string> parameterNames;

	mat data;
	unsigned int dimension = 0;
	double crowdingFactor = 0;


public:

	unsigned int numberOfDoESamples = 0;

	void setFileName(string);
	void setDimension(unsigned int);
	void setData(mat);
	void setParameterNames(vector<string> names);

	void reset(void);

	mat getData(void) const;
	vec getObjectiveFunctionValues(void) const;
	vec getFeasibilityValues(void) const;
	double getCrowdingFactor(void) const;

	vector<string> setHeader(void) const;
	void addConstraintName(const std::string& name) ;
	void setObjectiveFunctionName(string);

	void saveOptimizationHistoryFile(void);

	void updateOptimizationHistory(Design d);

	double calculateInitialImprovementValue(void) const;
	void print(void) const;

	void calculateCrowdingFactor(void);

};


} /* Namespace Rodop */

#endif
