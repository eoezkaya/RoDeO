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

#ifndef MULTI_LEVEL_MODEL
#define MULTI_LEVEL_MODEL

#include<string>
#include "surrogate_model.hpp"


class MultiLevelModel : public SurrogateModel {

private:

	std::string label;
	std::string labelErrorModel;
	std::string labelLowFidelityModel;

	std::string inputFileNameLowFidelityData;
	std::string inputFileNameHighFidelityData;
	std::string inputFileNameError;

	SurrogateModel *lowFidelityModel;
	SurrogateModel *errorModel;

	mat rawDataHighFidelity;
	mat rawDataHighFidelityForGammaTraining;
	mat rawDataHighFidelityForGammaTest;
	unsigned int NHiFi = 0;

	mat rawDataLowFidelity;
	unsigned int NLoFi = 0;

	mat rawDataError;
	mat rawDataErrorForGammaTraining;
	mat rawDataErrorForGammaTest;

	unsigned int dimHiFi = 0;
	unsigned int dimLoFi = 0;

	double gamma = 1.0;
	unsigned int maxIterationsForGammaTraining = 1000;



public:

	bool ifLowFidelityModelIsSet = false;
	bool ifErrorModelIsSet = false;
	bool ifErrorDataIsSet = false;
	bool ifHighFidelityDataHasGradients = false;
	bool ifLowFidelityDataHasGradients = false;

	MultiLevelModel(std::string);

	void readData(void);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;

	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);

	void updateAuxilliaryFields(void);
	void train(void);
	void trainGamma(void);

	double interpolate(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;
	void addNewSampleToData(rowvec newsample);

	void readHighFidelityData(void);
	void readLowFidelityData(void);

	void setDimensionsHiFiandLowFiModels(void);

	void prepareErrorData(void);
	void prepareTrainingDataForGammaOptimization(void);
	unsigned int findIndexHiFiToLowFiData(unsigned int indexHiFiData) const;


	void setinputFileNameHighFidelityData(std::string);
	void setinputFileNameLowFidelityData(std::string);

	void setParameterBounds(vec, vec);
	void setLowFidelityModel(std::string);
	void setErrorModel(std::string);

	mat getRawDataHighFidelity(void) const;
	mat getRawDataLowFidelity(void) const;
	mat getRawDataError(void) const;
	mat getRawDataHighFidelityForGammaTraining(void) const;
	mat getRawDataHighFidelityForGammaTest(void) const;

	double evaluateBrigdeFunction(rowvec x);

};



#endif
