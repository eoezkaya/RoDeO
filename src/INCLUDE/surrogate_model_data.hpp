/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#ifndef SURROGATE_MODEL_DATA_HPP
#define SURROGATE_MODEL_DATA_HPP

#include "output.hpp"
#include "bounds.hpp"
#include<string>

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;
using std::string;

class SurrogateModelData{

protected:

	unsigned int numberOfSamples = 0;
	unsigned int numberOfTestSamples = 0;
	unsigned int dimension = 0;


	mat rawData;
	mat X;
	mat Xraw;
	mat gradient;
	vec y;

	vec directionalDerivatives;
	mat differentiationDirections;

	mat XrawTest;
	mat XTest;
	vec yTest;

	double scalingFactorOutput = 1.0;

	OutputDevice outputToScreen;
	Bounds boxConstraints;


public:

	bool ifTestDataHasFunctionValues = false;
	bool ifDataHasGradients = false;
	bool ifDataHasDirectionalDerivatives = false;
	bool ifDataIsNormalized = false;
	bool ifDataIsRead = false;
	bool ifTestDataIsRead = false;

	SurrogateModelData();

	void reset(void);


	void setDisplayOn(void);
	void setDisplayOff(void);
	void setGradientsOn(void);
	void setGradientsOff(void);
	void setDirectionalDerivativesOn(void);
	void setDirectionalDerivativesOff(void);



	void setBoxConstraints(Bounds);
	void setBoxConstraintsFromData(void);
	Bounds getBoxConstraints(void) const;


	void assignDimensionFromData(void);
	void assignSampleInputMatrix(void);
	void assignSampleOutputVector(void);
	void assignGradientMatrix(void);
	void assignDifferentiationDirectionMatrix(void);
	void assignDirectionalDerivativesVector(void);

	void normalize(void);
	void normalizeSampleInputMatrix(void);
	void normalizeSampleInputMatrixTest(void);
	void normalizeDerivativesMatrix(void);
	void normalizeGradientMatrix(void);
	void normalizeOutputVector(void);

	bool isDataNormalized(void) const;
	bool isDataRead(void) const;


	unsigned int getNumberOfSamples(void) const;
	unsigned int getNumberOfSamplesTest(void) const;

	unsigned int getDimension(void) const;
	void setDimension(unsigned int);

	mat getRawData(void) const;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXTest(unsigned int index) const;
	rowvec getRowRawData(unsigned int index) const;
	rowvec getRowGradient(unsigned int index) const;
	rowvec getRowDifferentiationDirection(unsigned int index) const;

	mat getInputMatrix(void) const;

	rowvec getRowXRaw(unsigned int index) const;
	rowvec getRowXRawTest(unsigned int index) const;

	vec getOutputVector(void) const;
	void setOutputVector(vec);
	vec getOutputVectorTest(void) const;
	double getMinimumOutputVector(void) const;
	double getMaximumOutputVector(void) const;

	mat getGradientMatrix(void) const;
	vec getDirectionalDerivativesVector(void) const;

	double getScalingFactorForOutput(void) const;

	void readData(string);
	void readDataTest(string);


	void print(void) const;



};



#endif
