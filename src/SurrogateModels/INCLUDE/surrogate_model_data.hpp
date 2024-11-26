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

#ifndef SURROGATE_MODEL_DATA_HPP
#define SURROGATE_MODEL_DATA_HPP

#include "../../Bounds/INCLUDE/bounds.hpp"
#include "../../LinearAlgebra/INCLUDE/vector.hpp"
#include "../../LinearAlgebra/INCLUDE/matrix.hpp"
#include<string>
#include<vector>

using std::string;



namespace Rodop{

class SurrogateModelData{


protected:

	unsigned int numberOfSamples = 0;
	unsigned int numberOfTestSamples = 0;
	unsigned int numberOfValidationSamples = 0;
	unsigned int dimension = 0;

	string filenameFromWhichTrainingDataIsRead;

	mat rawData;
	mat rawDataWork;
	mat rawDataValidation;
	mat X;
	mat Xraw;

	mat gradient;
	mat gradientRaw;
	vec y;

	/* for tangent enhanced models only */
	vec directionalDerivatives;
	mat differentiationDirections;

	mat XrawTest;
	mat XTest;
	vec yTest;

	mat XrawValidation;
	mat XValidation;
	vec yValidation;

	Bounds boxConstraints;

	double ratioValidationData = 0.0;
	std::vector<int>indicesForValidationData;

public:

	bool ifTestDataHasFunctionValues = false;
	bool ifDataHasGradients = false;
	bool ifDataHasDirectionalDerivatives = false;
	bool ifDataIsNormalized = false;
	bool ifDataIsRead = false;
	bool ifTestDataIsRead = false;
	bool ifDisplay = false;

	SurrogateModelData();

	void reset(void);


	void setGradientsOn(void);
	void setGradientsOff(void);
	void setDirectionalDerivativesOn(void);
	void setDirectionalDerivativesOff(void);

	void setValidationRatio(double val);

	void setBoxConstraints(Bounds);
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

	bool isDataNormalized(void) const;
	bool isDataRead(void) const;

	void revertBackToFullData();

	unsigned int getNumberOfSamples(void) const;
	unsigned int getNumberOfSamplesValidation(void) const;
	unsigned int getNumberOfSamplesTest(void) const;

	unsigned int getDimension(void) const;
	void setDimension(unsigned int);

	mat getRawData(void) const;

	vec getRowX(unsigned int index) const;
	vec getRowXTest(unsigned int index) const;
	vec getRowXValidation(unsigned int index) const;


	vec getRowRawData(unsigned int index) const;

	vec getRowGradient(unsigned int index) const;
	vec getRowGradientRaw(unsigned int index) const;

	vec getRowDifferentiationDirection(unsigned int index) const;

	mat getInputMatrix(void) const;
	mat getInputMatrixTest(void) const;
	mat getInputMatrixValidation(void) const;

	vec getRowXRaw(unsigned int index) const;
	vec getRowXRawTest(unsigned int index) const;

	vec getOutputVector(void) const;
	void setOutputVector(vec);

	vec getOutputVectorTest(void) const;
	vec getOutputVectorValidation(void) const;

	double getMinimumOutputVector(void) const;
	double getMaximumOutputVector(void) const;

	mat getGradientMatrix(void) const;
	vec getDirectionalDerivativesVector(void) const;

	void readData(string);
	void readDataTest(string);

	std::vector<int> generateRandomIndicesForValidationData(int N, double ratio) const;



	void print(void) const;
	void printValidationSampleIndices(void) const;
	void printMessage(string);
	void printHere(const std::string& file = __FILE__, int line = __LINE__);

private:
	void checkDimensionAndNumberOfSamples();
};

}

#endif
