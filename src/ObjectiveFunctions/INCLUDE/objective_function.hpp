/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#ifndef OBJECTIVE_FUNCTION_HPP
#define OBJECTIVE_FUNCTION_HPP

#include <fstream>

#include "../../SurrogateModels/INCLUDE/kriging_training.hpp"
#include "../../SurrogateModels/INCLUDE/surrogate_model.hpp"
#include "../../LinearAlgebra/INCLUDE/vector.hpp"
#include "../../Design/INCLUDE/design.hpp"


#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif


namespace Rodop{


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


	std::string nameHighFidelityTrainingData;

	/* These are required only for multi-level option */

	bool ifMultiLevel = false;
	bool ifDefined = false;
	bool doesUseUDF = false; // User defined Function



	SURROGATE_MODEL modelHiFi  = ORDINARY_KRIGING;

	ObjectiveFunctionDefinition();
	bool checkIfDefinitionIsOk(void) const;

	void print(void) const;

	void printHighFidelityModel() const;

	std::string toString() const;
	string getNameOfSurrogateModel(SURROGATE_MODEL) const;
};


class ObjectiveFunction{


private:

	void bindWithOrdinaryKrigingModel();
	void bindWithUniversalKrigingModel();

	bool isHiFiEvaluation(void) const;
	void bindSurrogateModelSingleFidelity();


	double (*objectiveFunctionPtr)(const double*) = nullptr;
	void checkEvaluationModeForPrimalExecution() const;

protected:

	bool doesObjectiveFunctionPtrExist = false;
	std::string evaluationMode;
	std::string addDataMode;

	ObjectiveFunctionDefinition definition;
	Bounds boxConstraints;
	KrigingModel surrogateModel;
	SurrogateModel *surrogate;

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

	void validateEvaluationMode(const std::string& mode);
	void validateDataAddMode(const std::string& mode);

	void setEvaluationMode(const std::string& mode);
	void setDataAddMode(const std::string& mode);

	void setParametersByDefinition(ObjectiveFunctionDefinition);
	void bindSurrogateModel(void);

	void initializeSurrogate(void);
	void trainSurrogate(void);
	void printSurrogate(void) const;


	SURROGATE_MODEL getSurrogateModelType(void) const;

	void setParameterBounds(Bounds );

	void setNumberOfTrainingIterationsForSurrogateModel(unsigned int);

	void setDimension(unsigned int dimension);

	void setFileNameReadInput(std::string fileName);

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
	void calculateExpectedImprovementUsingDerivatives(DesignForBayesianOptimization &designCalculated) const;
	void calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const;
	void calculateSurrogateEstimate(DesignForBayesianOptimization &designCalculated) const;
	void calculateSurrogateEstimateUsingDerivatives(DesignForBayesianOptimization &designCalculated) const;


	void evaluateDesign(Design &d);
	void evaluateObjectiveFunction(void) const;

	double evaluateObjectiveFunctionDirectly(const Rodop::vec &x);

	void writeDesignVariablesToFile(Design &d) const;

	vec readOutput(const std::string &filename, unsigned int howMany) const;

	void readOutputDesign(Design &) const;


	void addDesignToData(Design &d);

	void addDesignToData(Design &d, string how);

	double interpolate(Rodop::vec x) const;
	double interpolateUsingDerivatives(Rodop::vec x) const;
	pair<double, double> interpolateWithVariance(Rodop::vec x) const;

	void print(void) const;
	std::string toString() const;
	void printInfoToLog(const string &msg) const;
	void printErrorToLog(const string &msg) const;
	void printWarningToLog(const string &msg) const;

	string generateOutputString(void) const;

	std::string getExecutionCommand(const std::string& exename) const;

	void removeVeryCloseSamples(const Design& globalOptimalDesign);
	void removeVeryCloseSamples(const Design& globalOptimalDesign, std::vector<Rodop::vec> samples);

	void setSigmaFactor(double);

	void setGlobalOptimalDesign(Design d);

	void setFunctionPtr(ObjectiveFunctionPtr func);


	double pdf(double x, double m, double s) const;
	double cdf(double x0, double mu, double sigma) const;


	std::string  generateFormattedString(std::string msg, char c, int totalLength) const;

};



} /*Namespace Rodop */
#endif
