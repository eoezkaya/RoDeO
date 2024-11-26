#ifndef DESIGN_HPP
#define DESIGN_HPP

#include <vector>
#include <string>
#include "../../LinearAlgebra/INCLUDE/vector.hpp"
#include "../../LinearAlgebra/INCLUDE/matrix.hpp"
#include "../../INCLUDE/globals.hpp"


using namespace std;

namespace Rodop{


class DesignForBayesianOptimization{

public:
	unsigned int dim;
	double objectiveFunctionValue;
	double valueAcquisitionFunction;

	double sigma;
	vec dv;

	vec constraintValues;
	vec constraintSigmas;
	vec constraintFeasibilityProbabilities;

	string tag = "DesignForBayesianOptimization";

	DesignForBayesianOptimization();
	DesignForBayesianOptimization(int dimension, int numberOfConstraints);
	DesignForBayesianOptimization(int dimension);
	DesignForBayesianOptimization(const vec& designVector, int numberOfConstraints);
	DesignForBayesianOptimization(const vec& designVector);

	void generateRandomDesignVector(void);
	void generateRandomDesignVector(vec lb, vec ub);

	void generateRandomDesignVectorAroundASample(const vec& sample, const vec& lb, const vec& ub, double factor = 0.01);


	double calculateProbalityThatTheEstimateIsLessThanAValue(double value);
	double calculateProbalityThatTheEstimateIsGreaterThanAValue(double value);

	double pdf(double x, double m, double s);
	double cdf(double x0, double mu, double sigma);

	void updateAcqusitionFunctionAccordingToConstraints(void);


	void gradientUpdateDesignVector(const vec &, const vec &, const vec &, double);
	void print(void) const;
	std::string toString() const;
	void reset(void);


};




class Design{

public:

	unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;

	vec designParameters;
	vec designParametersNormalized;
	vec constraintTrueValues;
	vec constraintEstimates;
	vec gradient;
	vec gradientLowFidelity;
	vec tangentDirection;

	double trueValue = 0.0;
	double estimatedValue = 0.0;
	double trueValueLowFidelity = 0;
	double tangentValue = 0.0;
	double tangentValueLowFidelity = 0.0;
	double improvementValue = 0.0;

	double ExpectedImprovementvalue = 0.0;

	vector<vec> constraintGradients;
	vector<vec> constraintGradientsLowFidelity;
	vector<vec> constraintTangentDirection;


	mat constraintGradientsMatrix;
	mat constraintGradientsMatrixLowFi;
	mat constraintDifferentiationDirectionsMatrix;
	mat constraintDifferentiationDirectionsMatrixLowFi;




	vec constraintTangent;
	vec constraintTangentLowFidelity;
	vec constraintTrueValuesLowFidelity;

	double surrogateEstimate = 0.0;
	vec constraintSurrogateEstimates;

	int ID = 0;
	string tag = "Design";

	bool isDesignFeasible = true;

	Design();
	Design(vec);
	Design(int);

	void setNumberOfConstraints(unsigned int howManyConstraints);

	void print(void) const;
	std::string toString() const;
	string generateOutputString(void) const;

	void saveToAFile(string) const;


	void saveDesignVectorAsCSVFile(const std::string& fileName) const;
	void saveDesignVector(const std::string& fileName) const;

	void generateRandomDesignVector(vec lb, vec ub);
	void generateRandomDesignVector(double lb, double ub);

	void generateRandomDifferentiationDirection(void);

	void setDimension(unsigned int);


	bool checkIfHasNan(void) const;

	void calculateNormalizedDesignParameters(vec lb, vec ub);
	void calculateDesignParametersFromNormalized(vec lb, vec ub);


	vec constructSampleObjectiveFunction(void) const;
	vec constructSampleObjectiveFunctionLowFi(void) const;
	vec constructSampleObjectiveFunctionWithGradient(void) const;
	vec constructSampleObjectiveFunctionWithZeroGradient(void) const;
	vec constructSampleObjectiveFunctionWithGradientLowFi(void) const;
	vec constructSampleObjectiveFunctionWithTangent(void) const;
	vec constructSampleObjectiveFunctionWithTangentLowFi(void) const;

	vec constructSampleConstraint(unsigned int constraintID) const;
	vec constructSampleConstraintLowFi(unsigned int constraintID) const;
	vec constructSampleConstraintWithTangent(unsigned int constraintID) const;
	vec constructSampleConstraintWithTangentLowFi(unsigned int constraintID) const;
	vec constructSampleConstraintWithGradient(unsigned int constraintID) const;
	vec constructSampleConstraintWithGradientLowFi(unsigned int constraintID) const;

	void reset(void);
	std::string generateFormattedString(std::string msg, char c, int totalLength) const;

};

} /* Namespace Rodop */


#endif
