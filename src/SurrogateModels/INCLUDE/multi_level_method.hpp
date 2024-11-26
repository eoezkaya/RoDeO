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
#ifndef MULTI_LEVEL_MODEL
#define MULTI_LEVEL_MODEL

#include<string>
#include "surrogate_model.hpp"
#include "kriging_training.hpp"


using std::string;


namespace Rodop{

class MultiLevelModel : public SurrogateModel{




private:

	string inputFileNameError;

	string filenameDataInputLowFidelity;

	SurrogateModel *lowFidelityModel;
	SurrogateModel *errorModel;

	KrigingModel surrogateModelKrigingLowFi;

	KrigingModel surrogateModelKrigingError;



	SURROGATE_MODEL modelIDHiFi;
	SURROGATE_MODEL modelIDLowFi;
	SURROGATE_MODEL modelIDError;


	SurrogateModelData dataLowFidelity;


	mat rawDataHighFidelity;
	mat XHighFidelity;


	mat rawDataLowFidelity;
	mat XLowFidelity;

	unsigned int numberOfSamplesLowFidelity = 0;


	mat rawDataError;

	Bounds boxConstraints;

	double alpha = 1.0;


	bool checkifModelIDIsValid(SURROGATE_MODEL id) const;
	void bindLowFidelityModel(void);
	void bindErrorModel(void);

public:


	bool ifSurrogateModelsAreSet = false;
	bool ifErrorDataIsSet = false;
	bool ifBoxConstraintsAreSet = false;

	void setName(std::string);
	void setDimension(unsigned int dim);


	void setNameOfInputFile(string filename);
	void setNameOfInputFileError(void);

	void setWriteWarmStartFileFlag(bool);
	void setReadWarmStartFileFlag(bool);



	void setNameOfHyperParametersFile(string filename);
	void setNumberOfTrainingIterations(unsigned int);

	void setinputFileNameHighFidelityData(string);
	void setinputFileNameLowFidelityData(string);

	void setBoxConstraints(Bounds boxConstraintsInput);

	void readData(void);
	void normalizeData(void);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;

	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);

	void updateAuxilliaryFields(void);

	void setIDHiFiModel(SURROGATE_MODEL);
	void setIDLowFiModel(SURROGATE_MODEL);


	void setDisplayOn(void);
	void setDisplayOff(void);

	unsigned int getNumberOfLowFiSamples(void) const;
	unsigned int getNumberOfHiFiSamples(void) const;

	void train(void);
	void trainLowFidelityModel(void);
	void trainErrorModel(void);


	void determineGammaBasedOnData(void);
	void determineAlpha(void);

	double getAlpha(void) const;


	double interpolate(vec x) const ;
	double interpolateUsingDerivatives(vec x) const;
	double interpolateLowFi(vec x) const;
	double interpolateError(vec x) const;

	void interpolateWithVariance(vec xp,double *f_tilde,double *ssqr) const;

	void addNewSampleToData(vec newsample);
	void addNewLowFidelitySampleToData(vec newsample);

	void updateModelWithNewData(void);

	void setDimensionsHiFiandLowFiModels(void);

	void prepareAndReadErrorData(void);

	unsigned int findIndexHiFiToLowFiData(unsigned int indexHiFiData) const;


	void bindModels(void);

	mat getRawDataHighFidelity(void) const;
	mat getRawDataLowFidelity(void) const;
	mat getRawDataError(void) const;


	void setNumberOfThreads(unsigned int);

};


} /* Namespace Rodop */


#endif
