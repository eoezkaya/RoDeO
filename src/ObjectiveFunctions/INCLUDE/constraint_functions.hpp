#ifndef CONSTRAINT_FUNCTIONS_HPP
#define CONSTRAINT_FUNCTIONS_HPP


#include "./objective_function.hpp"
#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../Design/INCLUDE/design.hpp"


namespace Rodop{

typedef double (*FunctionPtr)(double*);

class ConstraintDefinition{

public:

	std::string inequalityType;
	std::string constraintName;
	int ID = -1;
	double value = 0.0;

	void setDefinition(const std::string& definition);
	void print(void) const;

	std::string toString() const;
	std::string removeSpacesFromString(std::string inputString) const;


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

public:

	ConstraintFunction();

	double interpolate(Rodop::vec x) const;
	pair<double, double> interpolateWithVariance(Rodop::vec x) const;

	bool isUserDefinedFunction(void) const;

	void setConstraintDefinition(ConstraintDefinition);

	void setInequalityType(const std::string &type);
	std::string getInequalityType(void) const;

	void setInequalityTargetValue(double);
	double getInequalityTargetValue(void) const;

	bool checkFeasibility(double value) const;

	void setID(int givenID);
	int getID(void) const;

	void readOutputDesign(Design &d) const;

	void evaluateDesign(Design &d);
	void evaluateExplicitFunction(Design &d);
	void validateDesignParameters(const Design &d) const;
	void evaluateObjectiveDirectly(Design &d);
	void validateConstraintID(const Design &d) const;



	void addDesignToData(Design &d);

	double callUserDefinedFunction(Rodop::vec &x) const;

	void print(void) const;
	std::string toString() const;
	string generateOutputString(void) const;

	void trainSurrogate(void);

	void setUseExplicitFunctionOn(void);



};

} /*Namespace Rodop */

#endif
