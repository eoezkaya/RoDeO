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
#include "kriging_training.hpp"
#include "aggregation_model.hpp"

using std::string;


class MultiLevelModel : public SurrogateModel {

private:

	string inputFileNameLowFidelityData;
	string inputFileNameHighFidelityData;
	string inputFileNameError;

	SurrogateModel *lowFidelityModel;
	SurrogateModel *errorModel;

	KrigingModel surrogateModelKrigingLowFi;
	AggregationModel surrogateModelAggregationLowFi;

	KrigingModel surrogateModelKrigingError;
	AggregationModel surrogateModelAggregationError;


	mat rawDataHighFidelity;
	mat XHighFidelity;


	mat rawDataLowFidelity;
	mat XLowFidelity;

	unsigned int NLoFi = 0;
	unsigned int NHiFi = 0;

	mat rawDataError;


	unsigned int dimHiFi = 0;
	unsigned int dimLoFi = 0;

	double gamma = 1.0;
	unsigned int maxIterationsForGammaTraining = 1000;



public:

	bool ifInputFileNameForHiFiModelIsSet = false;
	bool ifInputFileNameForLowFiModelIsSet = false;

	bool ifLowFidelityModelIsSet = false;
	bool ifErrorModelIsSet = false;
	bool ifErrorDataIsSet = false;


	bool ifHighFidelityDataHasGradients = false;
	bool ifLowFidelityDataHasGradients = false;


	MultiLevelModel();
	MultiLevelModel(string);

	void setNameOfInputFile(string filename);
	void setNameOfInputFileError(void);

	void setNameOfHyperParametersFile(string filename);
	void setNumberOfTrainingIterations(unsigned int);

	void setinputFileNameHighFidelityData(string);
	void setinputFileNameLowFidelityData(string);

	void readData(void);
	void normalizeData(void);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;

	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);

	void updateAuxilliaryFields(void);


	void setGradientsOnLowFi(void);
	void setGradientsOnHiFi(void);
	void setGradientsOffLowFi(void);
	void setGradientsOffHiFi(void);

	void setDisplayOn(void);
	void setDisplayOff(void);



	void train(void);
	void trainLowFidelityModel(void);
	void trainErrorModel(void);


	void determineGammaBasedOnData(void);

	double interpolate(rowvec x) const ;
	double interpolateLowFi(rowvec x) const;
	double interpolateError(rowvec x) const;

	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;
	void addNewSampleToData(rowvec newsample);

	void readHighFidelityData(void);
	void readLowFidelityData(void);

	void setDimensionsHiFiandLowFiModels(void);

	void prepareErrorData(void);

	unsigned int findIndexHiFiToLowFiData(unsigned int indexHiFiData) const;


	void bindLowFidelityModel(void);
	void bindErrorModel(void);

	mat getRawDataHighFidelity(void) const;
	mat getRawDataLowFidelity(void) const;
	mat getRawDataError(void) const;


	unsigned int findNearestNeighbourLowFidelity(rowvec x) const;
	unsigned int findNearestNeighbourHighFidelity(rowvec x) const;

	double findNearestL1DistanceToALowFidelitySample(rowvec x) const;
	double findNearestL1DistanceToAHighFidelitySample(rowvec x) const;


};



#endif
