#ifndef OBJECTIVE_FUNCTION_HPP
#define OBJECTIVE_FUNCTION_HPP

#include <fstream>
#include <armadillo>
#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../SurrogateModels/INCLUDE/surrogate_model.hpp"
#include "../../SurrogateModels/INCLUDE/multi_level_method.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"
#include "../../Output/INCLUDE/output.hpp"

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif

typedef double (*ObjectiveFunctionPtr)(const double *);

class ObjectiveFunctionDefinition{

public:
	std::string name;
	std::string designVectorFilename;

	std::string executableName;
	std::string executableNameGradient;
	std::string executableNameDirectionalDerivative;

	std::string path;
	std::string outputFilename;
	std::string outputGradientFilename;

	/* These are required only for multi-level option */
	std::string executableNameLowFi;
	std::string executableNameLowFiGradient;
	std::string executableNameLowFiDirectionalDerivative;

	std::string pathLowFi;
	std::string outputFilenameLowFi;
	std::string outputFilenameLowFiGradient;

	std::string nameLowFidelityTrainingData;
	std::string nameHighFidelityTrainingData;

	bool ifMultiLevel = false;
	bool ifDefined = false;

	bool doesUseUDF = false;



	SURROGATE_MODEL modelHiFi  = ORDINARY_KRIGING;
	SURROGATE_MODEL modelLowFi = ORDINARY_KRIGING;

	ObjectiveFunctionDefinition();
	bool checkIfDefinitionIsOk(void) const;

	void print(void) const;

	void printHighFidelityModel() const;
	void printLowFidelityModel() const;
	string getNameOfSurrogateModel(SURROGATE_MODEL) const;
};


class ObjectiveFunction{

#ifdef UNIT_TESTS
	friend class ObjectiveFunctionTest;
	FRIEND_TEST(ObjectiveFunctionTest, constructor);
	FRIEND_TEST(ObjectiveFunctionTest, setParametersByDefinition);
	FRIEND_TEST(ObjectiveFunctionTest, setParameterBounds);
	FRIEND_TEST(ObjectiveFunctionTest, setDimensionAfterBindSurrogateModelCase1);
	FRIEND_TEST(ObjectiveFunctionTest, setNameAfterBindSurrogateModelCase1);
	FRIEND_TEST(ObjectiveFunctionTest, initializeSurrogateKriging);
	FRIEND_TEST(ObjectiveFunctionTest, initializeSurrogateGradientEnhanced);
	FRIEND_TEST(ObjectiveFunctionTest, initializeSurrogateTangentEnhanced);

	FRIEND_TEST(ObjectiveFunctionTest, evaluateGradient);


#endif


private:

	void bindWithOrdinaryKrigingModel();
	void bindWithUniversalKrigingModel();
	void bindWithGradientEnhancedModel();
	void bindWithTangentEnhancedModel();
	void bindWithMultiFidelityModel();

	bool isHiFiEvaluation(void) const;
	bool isLowFiEvaluation(void) const;
	void bindSurrogateModelSingleFidelity();

	void printWaitStatusIfSystemCallFails(int status) const;


	void evaluateGradient(void) const;


	double (*objectiveFunctionPtr)(const double*) = nullptr;


protected:

	bool doesObjectiveFunctionPtrExist = false;
	std::string evaluationMode;
	std::string addDataMode;

	ObjectiveFunctionDefinition definition;


	Bounds boxConstraints;


	KrigingModel surrogateModel;
	GeneralizedDerivativeEnhancedModel surrogateModelGradient;
	MultiLevelModel surrogateModelML;


	SurrogateModel *surrogate;

	OutputDevice output;

	unsigned int numberOfIterationsForSurrogateTraining = 10000;


	unsigned int dim = 0;

	double sampleMinimum = 0.0;

	double sigmaFactor = 1.0;






public:

	ObjectiveFunction();

	bool ifWarmStart = false;

	bool ifInitialized = false;
	bool ifParameterBoundsAreSet = false;
	bool ifDefinitionIsSet = false;
	bool ifSurrogateModelIsDefined = false;

	bool isMultiFidelityActive(void) const;



	void setEvaluationMode(std::string);
	void setDataAddMode(std::string mode);

	void setParametersByDefinition(ObjectiveFunctionDefinition);
	void bindSurrogateModel(void);

	void initializeSurrogate(void);
	void trainSurrogate(void);
	void printSurrogate(void) const;


	SURROGATE_MODEL getSurrogateModelType(void) const;
	SURROGATE_MODEL getSurrogateModelTypeLowFi(void) const;

	MultiLevelModel  getSurrogateModelML(void) const;


	void setDisplayOn(void);
	void setDisplayOff(void);

	void setParameterBounds(Bounds );

	void setNumberOfTrainingIterationsForSurrogateModel(unsigned int);

	void setDimension(unsigned int dimension);

	void setFileNameReadInput(std::string fileName);
	void setFileNameReadInputLowFidelity(std::string fileName);

	void setFileNameTrainingData(std::string fileName);


	void setExecutablePath(std::string);
	void setExecutableName(std::string);

	void setFileNameDesignVector(std::string);
	std::string getFileNameDesignVector(void) const;
	std::string getFileNameTrainingData(void) const;

	mat getTrainingData(void) const;
	std::string getName(void) const;


	void setFeasibleMinimum(double value);

	void calculateExpectedImprovement(DesignForBayesianOptimization &designCalculated) const;
	void calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const;
	void calculateSurrogateEstimate(DesignForBayesianOptimization &designCalculated) const;
	void calculateSurrogateEstimateUsingDerivatives(DesignForBayesianOptimization &designCalculated) const;


	void evaluateDesign(Design &d);
	void evaluateDesignGradient(Design &d);

	void evaluateObjectiveFunction(void);
	double evaluateObjectiveFunctionDirectly(const rowvec &x);

	void writeDesignVariablesToFile(Design &d) const;

	rowvec readOutput(string filename, unsigned int howMany) const;

	void readOutputDesign(Design &) const;


	void addDesignToData(Design &d);
	void addLowFidelityDesignToData(Design &d);

	void addDesignToData(Design &d, string how);

	bool checkIfGradientAvailable(void) const;
	double interpolate(rowvec x) const;
	double interpolateUsingDerivatives(rowvec x) const;
	pair<double, double> interpolateWithVariance(rowvec x) const;

	void print(void) const;
	string generateOutputString(void) const;

	std::string getExecutionCommand(const std::string& exename) const;

	void removeVeryCloseSamples(const Design& globalOptimalDesign);
	void removeVeryCloseSamples(const Design& globalOptimalDesign, std::vector<rowvec> samples);

	void setSigmaFactor(double);

	void setGlobalOptimalDesign(Design d);

	void setFunctionPtr(ObjectiveFunctionPtr func);


};


#endif
