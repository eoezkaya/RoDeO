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



using namespace arma;


class PartitionData{

public:

	std::string label;
	mat rawData;
	mat X;
	mat gradientData;

	vec yExact;
	vec ySurrogate;
	vec squaredError;

	PartitionData();
	PartitionData(std::string name);
	void fillWithData(mat);
	bool ifNormalized;
	bool ifHasGradientData;
	unsigned int numberOfSamples;
	unsigned int dim;
	void normalizeAndScaleData(vec xmin, vec xmax);
	double calculateMeanSquaredError(void) const;
	rowvec getRow(unsigned int indx) const;
	void saveAsCSVFile(std::string fileName);
	void print(void) const;

};

class SurrogateModel{

protected:
	unsigned int dim;
	unsigned int N;

	mat rawData;
	mat X;
	mat gradientData;
	vec y;


	std::string label;
	std::string hyperparameters_filename;
	std::string input_filename;


	double ymin,ymax,yave;
	vec xmin;
	vec xmax;

	bool ifInitialized;
	int numberOfHyperParameters;



public:

	SURROGATE_MODEL modelID;
	bool ifUsesGradientData;
	bool ifprintToScreen;

	SurrogateModel();
	SurrogateModel(std::string name);

	void ReadDataAndNormalize(void);

	virtual void initializeSurrogateModel(void) = 0;
	virtual void printSurrogateModel(void) const = 0;
	virtual void printHyperParameters(void) const = 0;
	virtual void saveHyperParameters(void) const = 0;
	virtual void loadHyperParameters(void) = 0;
	virtual void updateAuxilliaryFields(void);
	virtual void train(void) = 0;
	virtual double interpolate(rowvec x) const = 0;
	virtual double interpolateWithGradients(rowvec x) const = 0;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const = 0;

	double calculateInSampleError(void) const;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;
	std::string getInputFileName(void) const;

	void updateData(mat dataMatrix);
	void tryModelOnTestSet(PartitionData &testSet) const;
	void visualizeTestResults(void) const;



};


#endif
