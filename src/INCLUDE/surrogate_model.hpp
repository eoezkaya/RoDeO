/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */

#ifndef SURROAGE_MODEL_HPP
#define SURROAGE_MODEL_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "bounds.hpp"
#include "design.hpp"


using namespace arma;
using std::string;


class PartitionData{

public:

	std::string label;
	mat rawData;
	mat X;
	mat gradientData;

	vec yExact;
	vec ySurrogate;
	vec squaredError;

	bool ifHasGradientData = false;
	bool ifNormalized  = false;

	PartitionData();
	PartitionData(string name);
	void fillWithData(mat);

	unsigned int numberOfSamples = 0;
	unsigned int dim = 0;
	void normalizeAndScaleData(vec xmin, vec xmax);
	double calculateMeanSquaredError(void) const;
	rowvec getRow(unsigned int indx) const;
	void saveAsCSVFile(string fileName);
	void print(void) const;

};

class SurrogateModel{

protected:
	unsigned int dim = 0;
	unsigned int numberOfSamples = 0;
	unsigned int NTest = 0;

	mat rawData;
	mat X;
	mat Xraw;
	mat gradientData;
	vec y;
	vec yTest;
	mat XTestraw;
	mat XTest;

	std::string label;
	std::string hyperparameters_filename;
	std::string filenameDataInput;
	std::string filenameTestResults;


	double ymin,ymax,yave;

	Bounds boxConstraints;

	unsigned int numberOfHyperParameters  = 0;
	unsigned int numberOfTrainingIterations  = 10000;


public:

	bool ifInitialized = false;
	bool ifDataIsRead = false;
	bool ifNormalized = false;
	bool ifHasGradientData = false;
	bool ifHasTestData = false;
	bool ifDisplay = false;


	std::string modelName;


	mat testResults;

	SURROGATE_MODEL modelID;

	bool ifprintToScreen = false;
	bool ifPrintOutSampleError = false;

	SurrogateModel();
	SurrogateModel(std::string name);

	void readData(void);
	void normalizeData(void);

	void checkIfParameterBoundsAreOk(void) const;
	void checkRawData(void) const;

	void setParameterBounds(vec xmin, vec xmax);
	void setParameterBounds(double xmin, double xmax);
	void setParameterBounds(Bounds boxConstraintsInput);



	void setTestData(mat testData);

	string getNameOfHyperParametersFile(void) const;
	string getNameOfInputFile(void) const;

	unsigned int getDimension(void) const;
	unsigned int getNumberOfSamples(void) const;
	mat getRawData(void) const;
	vec getxmin(void) const;
	vec getxmax(void) const;


	virtual void setNameOfInputFile(string filename) = 0;
	virtual void setNameOfHyperParametersFile(string filename) = 0;
	virtual void setNumberOfTrainingIterations(unsigned int) = 0;

	virtual void initializeSurrogateModel(void) = 0;
	virtual void printSurrogateModel(void) const = 0;
	virtual void printHyperParameters(void) const = 0;
	virtual void saveHyperParameters(void) const = 0;
	virtual void loadHyperParameters(void) = 0;
	virtual void updateAuxilliaryFields(void);
	virtual void train(void) = 0;
	virtual double interpolate(rowvec x) const = 0;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const = 0;
	virtual void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const = 0;
	virtual void addNewSampleToData(rowvec newsample) = 0;


	double calculateInSampleError(void) const;
	void calculateOutSampleError(void);
	double getOutSampleErrorMSE(void) const;
	void saveTestResults(void) const;


	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;

	void tryModelOnTestSet(PartitionData &testSet) const;
	void visualizeTestResults(void) const;


	void printMsg(string msg) const;
	template <class T>
	void printMsg(string msg, T) const;



};


#endif
