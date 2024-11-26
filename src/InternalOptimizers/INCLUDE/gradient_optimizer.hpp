#ifndef GRADIENTOPTIMIZER_HPP
#define GRADIENTOPTIMIZER_HPP

#include <cfloat> // for FLT_MAX
#include <string>

#include "../../Bounds/INCLUDE/bounds.hpp"
#include "./general_purpose_optimizer.hpp"

namespace Rodop{


typedef double (*ObjectiveFunctionType)(double *);
typedef void (*GradientFunctionType)(double *, double *);


struct designPoint {
	vec x;
	vec gradient;
	double objectiveFunctionValue = 0;
	double L2NormGradient = 0;
} ;



class GradientOptimizer : public GeneralPurposeOptimizer{

private:

	std::string LineSearchMethod = "backtracking_line_search";
	std::string finiteDifferenceMethod;

	bool ifInitialPointIsSet = false;

	bool ifGradientFunctionIsSet = false;
	bool areFiniteDifferenceApproximationsToBeUsed = false;



	designPoint currentIterate;
	designPoint nextIterate;


	std::vector<designPoint> designParameterHistory;

	double tolerance = 10E-6;
	double epsilonForFiniteDifferences = 0.001;
	double maximumStepSize = 0.0;
	unsigned int maximumNumberOfIterationsInLineSearch  = 10;

	void (*calculateGradientFunction)(double *, double *);

	double optimalObjectiveFunctionValue = FLT_MAX;

	unsigned int numberOfMaximumFunctionEvaluations = 0;
	unsigned int numberOfFunctionEvaluations = 0;


	virtual vec calculateGradientFunctionInternal(const vec&);
	vec calculateCentralFiniteDifferences(designPoint &input);
	vec calculateForwardFiniteDifferences(designPoint &input);

public:

	GradientOptimizer();


	bool isOptimizationTypeMinimization(void) const;
	bool isOptimizationTypeMaximization(void) const;
	bool isInitialPointSet(void) const;

	bool isGradientFunctionSet(void) const;


	void setInitialPoint(const vec& input);
	void setMaximumNumberOfFunctionEvaluations(int);

	void setGradientFunction(GradientFunctionType functionToSet );

	void setMaximumStepSize(double);
	void setTolerance(double);
	void setMaximumNumberOfIterationsInLineSearch(int);
	void setFiniteDifferenceMethod(std::string);
	void setEpsilonForFiniteDifference(double);

	void evaluateObjectiveFunction(designPoint &);
	void evaluateGradientFunction(designPoint &);
	void approximateGradientUsingFiniteDifferences(designPoint &input);

	void checkIfOptimizationSettingsAreOk(void) const;


	void performGoldenSectionSearch(void);
	void performBacktrackingLineSearch(void);
	void performLineSearch(void);
	void optimize(void);

	double getOptimalObjectiveFunctionValue(void) const;

};

}

#endif
