#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP


#include "./surrogate_model.hpp"
#include "./linear_regression.hpp"
#include "../../LinearAlgebra/INCLUDE/linear_solver.hpp"
#include "../../InternalOptimizers/INCLUDE/ea_optimizer.hpp"
#include "../../CorrelationFunctions/INCLUDE/exponential_correlation_function.hpp"

#ifdef UNIT_TESTS
#include "gtest/gtest.h"
#endif


namespace Rodop{

enum KrigingObjectiveType { VALIDATION_ERROR, MAXIMUM_LIKELIHOOD };

class KrigingModel : public SurrogateModel{

#ifdef UNIT_TESTS
	friend class Kriging1DModelTest;
	FRIEND_TEST(Kriging1DModelTest, updateAuxilliaryFields);
	FRIEND_TEST(Kriging1DModelTest, calculateLikelihoodFailsDueToNonPositiveDefiniteR);
	friend class Kriging2DModelTest;
	FRIEND_TEST(Kriging2DModelTest, readDataWorks);
	FRIEND_TEST(Kriging2DModelTest, updateAuxilliaryFields);
	friend class  KrigingModelProfilingTest;
	FRIEND_TEST( KrigingModelProfilingTest,  measureTimeUpdateAuxFields);
	#endif

private:


	KrigingObjectiveType  objectiveFunctionType = MAXIMUM_LIKELIHOOD;
	/* Auxiliary vectors */
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	vec R_inv_ys;
	vec vectorOfOnes;


	CholeskySystem linearSystemCorrelationMatrix;
	//	SVDSystem      linearSystemCorrelationMatrixSVD;

	double beta0 = 0.0;
	double sigmaSquared = 0.0;

	bool ifUsesLinearRegression = false;
	bool ifCorrelationFunctionIsInitialized = false;

	LinearModel linearModel;

	ExponentialCorrelationFunction correlationFunction;

	void updateWithNewData(void);
	void updateModelParams(void);
	void calculateBeta0(void);
	void checkDimension();
	void checkIfDataIsRead();
	void checkFilenameHyperparameters() const;
	void checkIfDataIsNormalized();

public:

	bool ifUseAverageBeta0 = false;

	KrigingModel();


	void readData(void);
	void normalizeData(void);

	void setBoxConstraints(Bounds boxConstraintsInput);

	void setDimension(unsigned int);
	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);


	void setNumberOfTrainingIterations(unsigned int);


	void initializeSurrogateModel(void);

	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void logHyperParameters() const;

	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void setHyperParameters(vec);
	vec getHyperParameters(void) const;

	void train(void);

	double interpolate(vec x) const ;
	double interpolateUsingDerivatives(vec x) const ;
	void interpolateWithVariance(vec xp,double *f_tilde,double *ssqr) const;


	void addNewSampleToData(vec newsample);
	void addNewLowFidelitySampleToData(vec newsample);


	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);

	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);

	mat getCorrelationMatrix(void) const;

	void resetDataObjects(void);
	void resizeDataObjects(void);


	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);
	void updateAuxilliaryFieldsWithSVDMethod(void);
	void checkAuxilliaryFields(void) const;

	double calculateLikelihoodFunction(vec);
	double calculateValidationErrorForGivenHyperparameters(vec hyperParameters);

	void setObjectiveFunctionTypeForModelTraining(KrigingObjectiveType type);
};




class KrigingHyperParameterOptimizer : public EAOptimizer{

#ifdef UNIT_TESTS
	friend class KrigingHyperParameterOptimizerTest;
	FRIEND_TEST(KrigingHyperParameterOptimizerTest, initializeKrigingModelObject);
#endif

private:

	double calculateObjectiveFunctionInternal(const vec& input);
	KrigingModel  KrigingModelForCalculations;



public:

	KrigingObjectiveType  objectiveFunctionType = VALIDATION_ERROR;
	void initializeKrigingModelObject(KrigingModel);
	void printInternalObject() const;
	bool ifModelObjectIsSet = false;



};

} /* Namespace Rodop */




#endif
