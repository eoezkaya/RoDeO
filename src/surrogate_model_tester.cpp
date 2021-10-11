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



#include "surrogate_model_tester.hpp"
#include "auxiliary_functions.hpp"
#include <cassert>



SurrogateModelTester::SurrogateModelTester(){



}

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

}


void SurrogateModelTester::setSurrogateModel(SURROGATE_MODEL modelType){

	surrogateModelType = modelType;

	switch(modelType) {
	  case LINEAR_REGRESSION:

		 surrogateModel = &linearModel;

	    break;
	  case ORDINARY_KRIGING:

		  surrogateModel = &krigingModel;

		  break;
	  case UNIVERSAL_KRIGING:
		  krigingModel.setLinearRegressionOn();
		  surrogateModel = &krigingModel;


	    break;

	  case AGGREGATION:

		  surrogateModel = &aggregationModel;
		  surrogateModel->setGradientsOn();
		  break;

	  case MULTI_LEVEL:
		  multilevelModel.setinputFileNameLowFidelityData(fileNameTraingDataLowFidelity);
		  surrogateModel = &multilevelModel;
		  break;

	  default:

		  outputToScreen.printErrorMessageAndAbort("Unknown modelType for the surrogate model!");

	}

	assert(isNotEmpty(name));
	surrogateModel->setName(name);



	assert(isNotEmpty(fileNameTraingData));

	surrogateModel->setNameOfInputFile(fileNameTraingData);

	assert(isNotEmpty(fileNameTestData));
	surrogateModel->setNameOfInputFileTest(fileNameTestData);



	surrogateModel->setNameOfHyperParametersFile(name);



	ifSurrogateModelSpecified = true;
}


bool SurrogateModelTester::isSurrogateModelSpecified(void) const{

	return ifSurrogateModelSpecified;

}



void SurrogateModelTester::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());
	boxConstraints = boxConstraintsInput;
}

Bounds SurrogateModelTester::getBoxConstraints(void) const{

	return boxConstraints;

}



void SurrogateModelTester::performSurrogateModelTest(void){

	outputToScreen.printMessage("Performing surrogate model test...");



	surrogateModel->readData();
	surrogateModel->readDataTest();



	if(boxConstraints.areBoundsSet()){

		surrogateModel->setBoxConstraints(boxConstraints);

	}
	else{

		surrogateModel->setBoxConstraintsFromData();

	}


	surrogateModel->normalizeData();

	surrogateModel->normalizeDataTest();


	surrogateModel->initializeSurrogateModel();



	surrogateModel->setNumberOfTrainingIterations(numberOfTrainingIterations);
	surrogateModel->train();

	surrogateModel->tryOnTestData();




}

void SurrogateModelTester::setDisplayOn(void){


	outputToScreen.ifScreenDisplay = true;
	outputToScreen.printMessage("Setting display on for the surrogate model tester...");

	if(ifSurrogateModelSpecified){

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
