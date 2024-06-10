#ifndef CONSTRAINT_FUNCTIONS_HPP
#define CONSTRAINT_FUNCTIONS_HPP


#include <armadillo>
#include <cassert>

#include "./objective_function.hpp"
#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"
#include "../../../externalFunctions/INCLUDE/externalFunctions.hpp"

typedef double (*FunctionPtr)(double*);

class ConstraintDefinition{

public:

	std::string inequalityType;
	std::string constraintName;
	int ID = -1;
	double value = 0.0;


	void setDefinition(std::string definition);
	void print(void) const;



};



class ConstraintFunction: public ObjectiveFunction {

#ifdef UNIT_TESTS
	friend class ConstraintFunctionTest;
	FRIEND_TEST(ConstraintFunctionTest, useFunctionPointer);

#endif

private:

	ConstraintDefinition definitionConstraint;

	bool ifFunctionExplictlyDefined = false;
	double (*functionPtr)(double*) = NULL;


	std::vector<FunctionPtr> functionVector = { constraintFunction0,
			constraintFunction1,
			constraintFunction2,
			constraintFunction3};


public:

	ConstraintFunction();

	double interpolate(rowvec x) const;
	pair<double, double> interpolateWithVariance(rowvec x) const;

	bool isUserDefinedFunction(void) const;

	void setConstraintDefinition(ConstraintDefinition);

	void setInequalityType(std::string);
	std::string getInequalityType(void) const;

	void setInequalityTargetValue(double);
	double getInequalityTargetValue(void) const;

	bool checkFeasibility(double value) const;

	void setID(int givenID);
	int getID(void) const;

	void readOutputDesign(Design &d) const;

	void evaluateDesign(Design &d);
	void addDesignToData(Design &d);

	double callUserDefinedFunction(rowvec &x) const;

	void print(void) const;
	string generateOutputString(void) const;

	void trainSurrogate(void);

	void setUseExplicitFunctionOn(void);

};



#endif
