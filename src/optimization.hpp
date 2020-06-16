#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP


#include <armadillo>
#include "kriging_training.hpp"

using namespace arma;


class ObjectiveFunction{


private:
	std::string name;
	double (*objectiveFunPtr)(double *);
	KrigingModel surrogateModel;
	unsigned int dim;

public:
	ObjectiveFunction(std::string, double (*objFun)(double *), unsigned int);
	ObjectiveFunction();
	void trainSurrogate(void);

	double calculateExpectedImprovement(rowvec x);
	double evaluate(rowvec x);
	double ftilde(rowvec x) const;
	void print(void) const;

};




class ConstraintFunction{


private:
	unsigned int dim;
	std::string name;
	double (*pConstFun)(double *);
	double targetValue;
	std::string inequalityType;
	bool ifNeedsSurrogate;
	KrigingModel surrogateModel;
public:
	ConstraintFunction(std::string, std::string, double, double (*constFun)(double *), unsigned int dimension, bool ifNeedsSurrogate = false);
	ConstraintFunction();
	void trainSurrogate(void);

	bool checkFeasibility(double value);

	double calculateEI(rowvec x) const;
	double evaluate(rowvec x);
	double ftilde(rowvec x) const;
	void print(void) const;
};






class OptimizerWithGradients {
public:
	std::string name;
	unsigned int size_of_dv;
	unsigned int max_number_of_samples;
	unsigned int iterMaxEILoop;
	vec lower_bound_dv;
	vec upper_bound_dv;
	bool doesValidationFileExist = true;

	OptimizerWithGradients();
	OptimizerWithGradients(int);
	OptimizerWithGradients(std::string ,int );
	void print(void);
	void EfficientGlobalOptimization(void);


	double (*adj_fun)(double *, double *);

};


class Optimizer {

private:

	vec lowerBounds;
	vec upperBounds;
	mat optimizationHistory;


	std::vector<ConstraintFunction> constraintFunctions;
	ObjectiveFunction objFun;



public:

	std::string name;
	unsigned int dimension;
	unsigned int numberOfConstraints;
	unsigned int maxNumberOfSamples;
	unsigned int howOftenTrainModels;

	unsigned int iterGradientEILoop;
	std::string optimizationType;
	bool ifVisualize;

	double epsilon_EI;
	unsigned int iterMaxEILoop;

	Optimizer(std::string ,int, std::string);
	void print(void) const;
	void visualizeOptimizationHistory(void) const;
	void EfficientGlobalOptimization(void);
	void trainSurrogates(void);

	void setBoxConstraints(std::string filename="BoxConstraints.csv");
	void setBoxConstraints(double lb, double ub);
	void setBoxConstraints(vec lb, vec ub);

	void addConstraint(ConstraintFunction &constFunc);

	void evaluateConstraints(rowvec x, rowvec &constraintValues);

	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec constraintValues);

	void addObjectFunction(ObjectiveFunction &objFunc);


};



#endif
