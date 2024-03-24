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

#ifndef SURROAGE_MODEL_HPP
#define SURROAGE_MODEL_HPP
#include <armadillo>
#include "../../INCLUDE/Rodeo_macros.hpp"
#include "../../Bounds/INCLUDE/bounds.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"
#include "./surrogate_model_data.hpp"

using namespace arma;
using std::string;




class SurrogateModel{



protected:

	unsigned int dimension = 0;
	unsigned int numberOfSamples = 0;

	std::string name;

	std::string filenameHyperparameters;
	std::string filenameDataInput;
	std::string filenameDataInputTest;
	std::string filenameTestResults = "surrogateTestResults.csv";


	unsigned int numberOfHyperParameters  = 0;
	unsigned int numberOfTrainingIterations  = 10000;

	SurrogateModelData data;
	OutputDevice output;

	bool ifHasGradientData = false;


	bool ifWriteWarmStartFile = false;
	bool ifReadWarmStartFile  = false;

	unsigned int numberOfThreads = 1;

	unsigned int minimumNumberOfSamplesForTrainingDataShrinkage = 30;

	Bounds boxConstraints;


	double standardDeviationOfGeneralizationError = 0.0;


	vec sampleWeights;
	double targetForSampleWeights = 0.0;

	Design globalOptimalDesign;





public:

	bool ifInitialized = false;
	bool ifDataIsRead = false;
	bool ifTestDataIsRead = false;
	bool ifNormalized = false;
	bool ifModelTrainingIsDone = false;
	bool ifHasTestData = false;
	bool ifNormalizedTestData = false;


	bool ifVaryingSampleWeights = false;
	bool ifTargetForSampleWeightsIsSet = false;


	virtual void setDimension(unsigned int);

	mat testResults;

	double generalizationError = 0.0;

	SurrogateModel();

	virtual void setName(std::string);

	string getName(void) const;


	void readDataTest(void);
	void normalizeDataTest(void);



	void printData(void) const;

	void checkRawData(void) const;

	void setNumberOfThreads(unsigned int);

	virtual void setWriteWarmStartFileFlag(bool);
	virtual void setReadWarmStartFileFlag(bool);

	void setGradientsOn(void);
	void setGradientsOff(void);
	bool areGradientsOn(void) const;

	void setGlobalOptimalDesign(Design d);

	virtual void setDisplayOn(void);
	virtual void setDisplayOff(void);

	string getNameOfHyperParametersFile(void) const;
	string getNameOfInputFile(void) const;

	unsigned int getDimension(void) const;
	unsigned int getNumberOfSamples(void) const;
	mat getRawData(void) const;



	void setNameOfInputFileTest(string filename);
	void setNameOfOutputFileTest(string filename);


	virtual void setNameOfInputFile(string filename) = 0;
	virtual void setNameOfHyperParametersFile(string filename) = 0;
	virtual void setNumberOfTrainingIterations(unsigned int) = 0;

	virtual void setBoxConstraints(Bounds boxConstraintsInput) = 0;

	virtual void readData(void) = 0;
	virtual void normalizeData(void) = 0;
	virtual void initializeSurrogateModel(void) = 0;
	virtual void printSurrogateModel(void) const = 0;
	virtual void printHyperParameters(void) const = 0;
	virtual void saveHyperParameters(void) const = 0;
	virtual void loadHyperParameters(void) = 0;
	virtual void updateAuxilliaryFields(void);
	virtual void train(void) = 0;
	virtual double interpolate(rowvec x) const = 0;
	virtual double interpolateUsingDerivatives(rowvec x) const = 0;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const = 0;

	virtual void addNewSampleToData(rowvec newsample) = 0;
	virtual void addNewLowFidelitySampleToData(rowvec newsample) = 0;

	virtual void updateModelWithNewData(void) = 0;


	vec interpolateVector(mat X) const;

	void tryOnTestData(void);

	double calculateInSampleError(void) const;
	double calculateOutSampleError(void);

	void saveTestResults(void) const;

	mat getX(void) const;
	vec gety(void) const;
	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;

    unsigned int countHowManySamplesAreWithinBounds(vec lb, vec ub);
	void reduceTrainingData(vec lb, vec ub) const;

	void printGeneralizationError(void) const;

	void generateSampleWeights(void);
	void generateSampleWeightsAccordingToGlobalOptimum(void);
	void printSampleWeights(void) const;

	void removeVeryCloseSamples(const Design& globalOptimalDesign);
	void removeVeryCloseSamples(const Design& globalOptimalDesign , std::vector<rowvec> samples);

};


#endif
