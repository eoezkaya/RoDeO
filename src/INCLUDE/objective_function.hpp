/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
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
 * General Public License along with RoDEO.
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
#include <armadillo>
#include "kriging_training.hpp"
#include "tgek.hpp"
#include "aggregation_model.hpp"
#include "surrogate_model.hpp"
#include "multi_level_method.hpp"
#include "design.hpp"
#include "output.hpp"




class ObjectiveFunctionDefinition{

public:
	std::string name;
	std::string designVectorFilename;

	std::string executableName;
	std::string path;
	std::string outputFilename;

	/* These are required only for multi-level option */
	std::string executableNameLowFi;
	std::string pathLowFi;
	std::string outputFilenameLowFi;

	std::string nameLowFidelityTrainingData;
	std::string nameHighFidelityTrainingData;

	bool ifMultiLevel = false;
	bool ifDefined = false;


	SURROGATE_MODEL modelHiFi  = ORDINARY_KRIGING;
	SURROGATE_MODEL modelLowFi = ORDINARY_KRIGING;

	ObjectiveFunctionDefinition();
	bool checkIfDefinitionIsOk(void) const;
	unsigned int identifyCase(void) const;

	void print(void) const;


};


class ObjectiveFunction{


private:
	void readOnlyFunctionalValue(Design &d) const;
	void readFunctionalValueAndTangent(Design &d) const;
	void readFunctionalValueAndAdjoint(Design &d) const;
	void bindWithOrdinaryKrigingModel();
	void bindWithUniversalKrigingModel();
	void bindWithGradientEnhancedModel();
	void bindWithTangentEnhancedKrigingModel();
	void bindWithMultiFidelityModel();

	bool isHiFiEvaluation(void) const;
	bool isLowFiEvaluation(void) const;
	void bindSurrogateModelSingleFidelity();

protected:


	std::string evaluationMode;
	ObjectiveFunctionDefinition definition;


	Bounds boxConstraints;


	KrigingModel surrogateModel;
	GGEKModel surrogateModelGradient;
	MultiLevelModel surrogateModelML;
	TGEKModel       surrogateModelWithTangents;


	SurrogateModel *surrogate;

	OutputDevice output;

	unsigned int numberOfIterationsForSurrogateTraining = 10000;


	unsigned int dim = 0;

	double yMin = 0.0;

	double sigmaFactor = 0.5;



public:

	ObjectiveFunction();

	bool ifDoErequired = true;
	bool ifWarmStart = false;
	bool ifInitialized = false;
	bool ifParameterBoundsAreSet = false;
	bool ifDefinitionIsSet = false;
	bool ifSurrogateModelIsDefined = false;


	void setEvaluationMode(std::string);


	void bindSurrogateModel(void);

	void initializeSurrogate(void);
	void trainSurrogate(void);
	void printSurrogate(void) const;
	SURROGATE_MODEL getSurrogateModelType(void) const;

	KrigingModel     getSurrogateModel(void) const;
	GGEKModel getSurrogateModelGradient(void) const;
	MultiLevelModel  getSurrogateModelML(void) const;
	TGEKModel        getSurrogateModelTangent(void) const;


	void setGradientOn(void);
	void setGradientOff(void);

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setParameterBounds(Bounds );

	void setNumberOfTrainingIterationsForSurrogateModel(unsigned int);

	void setDimension(unsigned int dimension);
	unsigned int getDimension(void) const;

	std::string getName(void) const{
		return definition.name;
	}


	void setFileNameReadInput(std::string fileName);
	void setFileNameReadInputLowFidelity(std::string fileName);

	void saveDoEData(std::vector<rowvec>) const;
	void setExecutablePath(std::string);
	void setExecutableName(std::string);

	void setFileNameDesignVector(std::string);
	std::string getFileNameDesignVector(void) const;
	std::string getFileNameTrainingData(void) const;


	void setParametersByDefinition(ObjectiveFunctionDefinition);


	void calculateExpectedImprovement(DesignForBayesianOptimization &designCalculated) const;
	void calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const;

	void evaluateDesign(Design &d);
	void evaluateObjectiveFunction(void);

	void writeDesignVariablesToFile(Design &d) const;


	rowvec readOutput(unsigned int) const;
	void readOutputDesign(Design &) const;


	void addDesignToData(Design &d);
	void addLowFidelityDesignToData(Design &d);

	bool checkIfGradientAvailable(void) const;
	double interpolate(rowvec x) const;
	pair<double, double> interpolateWithVariance(rowvec x) const;

	void print(void) const;
	std::string getExecutionCommand(string, string) const;


	void reduceTrainingDataFiles(unsigned int, double) const;

	void setSigmaFactor(double);


};


#endif
