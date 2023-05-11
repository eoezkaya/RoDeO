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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */


#include "surrogate_model_tester.hpp"
#include "auxiliary_functions.hpp"
#include <cassert>



SurrogateModelTester::SurrogateModelTester(){}

unsigned int SurrogateModelTester::getDimension(void) const{
	return dimension;
}

void SurrogateModelTester::setDimension(unsigned int value){
	dimension = value;
}

void SurrogateModelTester::setName(string nameInput){
	assert(isNotEmpty(nameInput));

	name = nameInput;
}

void SurrogateModelTester::setNumberOfTrainingIterations(unsigned int nIterations) {
	numberOfTrainingIterations = nIterations;
	if(ifbindSurrogateModelisDone){
		surrogateModel->setNumberOfTrainingIterations(numberOfTrainingIterations);

	}
}

void SurrogateModelTester::setSurrogateModel(SURROGATE_MODEL modelType){

	surrogateModelType = modelType;
	ifSurrogateModelSpecified = true;

}

void SurrogateModelTester::setSurrogateModelLowFi(SURROGATE_MODEL modelType){

	surrogateModelTypeLowFi = modelType;
	ifSurrogateModelLowFiSpecified = true;

}

void SurrogateModelTester::bindSurrogateModels(void){

	assert(ifSurrogateModelSpecified);
	assert(isNotEmpty(fileNameTraingData));
	assert(isNotEmpty(name));
	assert(isNotEmpty(fileNameTestData));
	assert(dimension>0);

	if(!ifMultiLevel){

		if(surrogateModelType == LINEAR_REGRESSION ){
			linearModel.setDimension(dimension);
			surrogateModel = &linearModel;
		}
		if(surrogateModelType == ORDINARY_KRIGING){
			krigingModel.setDimension(dimension);
			surrogateModel = &krigingModel;
		}
		if(surrogateModelType == UNIVERSAL_KRIGING){
			krigingModel.setLinearRegressionOn();
			krigingModel.setDimension(dimension);
			surrogateModel = &krigingModel;
		}
		if(surrogateModelType == TANGENT){
			tangentModel.setDimension(dimension);
			surrogateModel = &tangentModel;
		}
		if(surrogateModelType == GRADIENT_ENHANCED){
			generalizedGradientEnhancedModel.setDimension(dimension);
			surrogateModel = &generalizedGradientEnhancedModel;
		}



		surrogateModel->setNameOfInputFile(fileNameTraingData);


	}

	else{

		assert(ifSurrogateModelLowFiSpecified);
		multilevelModel.setIDHiFiModel(surrogateModelType);
		multilevelModel.setIDLowFiModel(surrogateModelTypeLowFi);
		multilevelModel.setinputFileNameHighFidelityData(fileNameTraingData);
		multilevelModel.setinputFileNameLowFidelityData(fileNameTraingDataLowFidelity);
		multilevelModel.bindModels();
		multilevelModel.setDimension(dimension);
		multilevelModel.setName(name);


		surrogateModel = &multilevelModel;


	}

	surrogateModel->setNameOfInputFileTest(fileNameTestData);

	ifbindSurrogateModelisDone = true;

}

void SurrogateModelTester::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());
	boxConstraints = boxConstraintsInput;
}

Bounds SurrogateModelTester::getBoxConstraints(void) const{
	return boxConstraints;
}

void SurrogateModelTester::performSurrogateModelTest(void){

	assert(boxConstraints.areBoundsSet());
	assert(ifbindSurrogateModelisDone);

	outputToScreen.printMessage("Performing surrogate model test...");
	surrogateModel->setBoxConstraints(boxConstraints);

	outputToScreen.printMessage("Reading training data...");
	surrogateModel->readData();
	outputToScreen.printMessage("Reading test data...");
	surrogateModel->readDataTest();

	surrogateModel->setBoxConstraints(boxConstraints);
	outputToScreen.printMessage("Data normalization...");
	surrogateModel->normalizeData();
	surrogateModel->normalizeDataTest();
	outputToScreen.printMessage("Surrogate model initialization...");
	surrogateModel->initializeSurrogateModel();
	surrogateModel->setNumberOfTrainingIterations(numberOfTrainingIterations);
	outputToScreen.printMessage("Surrogate model training...");
	surrogateModel->train();

	if(outputToScreen.ifScreenDisplay){

		surrogateModel->setDisplayOn();
	}

	surrogateModel->tryOnTestData();
	surrogateModel->saveTestResults();

	surrogateModel->printGeneralizationError();

}

void SurrogateModelTester::setDisplayOn(void){


	outputToScreen.ifScreenDisplay = true;
	outputToScreen.printMessage("Setting display on for the surrogate model tester...");

	if(ifbindSurrogateModelisDone){

		outputToScreen.printMessage("Setting display on for the surrogate model...");
		surrogateModel->setDisplayOn();

	}


}

void SurrogateModelTester::setDisplayOff(void){


	outputToScreen.ifScreenDisplay = true;

	if(ifSurrogateModelSpecified){

		surrogateModel->setDisplayOff();

	}
}



void SurrogateModelTester::setFileNameTrainingData(string filename){

	assert(isNotEmpty(filename));
	fileNameTraingData = filename;

}


void SurrogateModelTester::setFileNameTrainingDataLowFidelity(string filename){

	assert(isNotEmpty(filename));
	fileNameTraingDataLowFidelity = filename;

}




void SurrogateModelTester::setFileNameTestData(string filename){

	assert(isNotEmpty(filename));
	fileNameTestData = filename;

}

void SurrogateModelTester::print(void) const{

	outputToScreen.printMessage("\n\nSurrogate model test information...");
	outputToScreen.printMessage("Dimension = ", dimension);
	outputToScreen.printMessage("Training data file name = ", fileNameTraingData);
	outputToScreen.printMessage("Test data file name = ", fileNameTestData);
	outputToScreen.printMessage("Surrogate model ID = ", surrogateModelType);



}
