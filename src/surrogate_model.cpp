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
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "surrogate_model.hpp"
#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"



#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;



SurrogateModel::SurrogateModel(){


}

SurrogateModel::SurrogateModel(std::string nameInput){


	setName(nameInput);

	filenameDataInput = name +".csv";
	filenameTestResults = name + "_TestResults.csv";

}

void SurrogateModel::setName(std::string nameInput){

	assert(isNotEmpty(nameInput));

	name = nameInput;

}



void SurrogateModel::setGradientsOn(void){

	data.setGradientsOn();
	ifHasGradientData = true;


}

void SurrogateModel::setGradientsOff(void){

	data.setGradientsOff();
	ifHasGradientData = false;

}

bool SurrogateModel::areGradientsOn(void) const{

	return ifHasGradientData;

}


void SurrogateModel::setDisplayOn(void){

	data.setDisplayOn();
	output.ifScreenDisplay = true;

}

void SurrogateModel::setDisplayOff(void){

	data.setDisplayOff();
	output.ifScreenDisplay = false;

}


void SurrogateModel::checkIfParameterBoundsAreOk(void) const{

	assert(boxConstraints.checkIfBoundsAreValid());


}


void SurrogateModel::setParameterBounds(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	data.setBoxConstraints(boxConstraintsInput);

	boxConstraints = boxConstraintsInput;


}



void SurrogateModel::setParameterBounds(vec xmin, vec xmax){

	Bounds boxConstraints(xmin,xmax);
	setParameterBounds(boxConstraints);

}

void SurrogateModel::setParameterBounds(double xmin, double xmax){

	Bounds boxConstraints(data.getDimension());
	boxConstraints.setBounds(xmin,xmax);

	setParameterBounds(boxConstraints);
}


void SurrogateModel::setBoxConstraintsFromData(void){

	data.setBoxConstraintsFromData();

	setParameterBounds(data.getBoxConstraints());


}

std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return hyperparameters_filename;

}




std::string SurrogateModel::getNameOfInputFile(void) const{

	return this->filenameDataInput;

}


unsigned int SurrogateModel::getDimension(void) const{

	return data.getDimension();


}

unsigned int SurrogateModel::getNumberOfSamples(void) const{

	return data.getNumberOfSamples();


}

mat SurrogateModel::getRawData(void) const{

	return data.getRawData();


}



void SurrogateModel::readData(void){


	data.readData(filenameDataInput);

	ifDataIsRead = true;

}

void SurrogateModel::readDataTest(void){

	data.readDataTest(filenameDataInputTest);

}



void SurrogateModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalizeSampleInputMatrix();

	ifNormalized = true;

}

void SurrogateModel::normalizeDataTest(void){


	data.normalizeSampleInputMatrixTest();


}


void SurrogateModel::updateAuxilliaryFields(void){




}




void SurrogateModel::setTestData(mat testData){

	assert(testData.n_rows > 0);
	assert(testData.n_cols == data.getDimension() +1);
	assert(boxConstraints.areBoundsSet());

	NTest = testData.n_rows;

	XTestraw = testData.submat(0,0,NTest-1, data.getDimension()-1);

	yTest = testData.col(data.getDimension());

	XTest = normalizeMatrix(XTestraw, this->boxConstraints);

	XTest = (1.0/data.getDimension())*XTest;

	ifHasTestData = true;
}




double SurrogateModel::calculateInSampleError(void) const{

	assert(ifInitialized);

	double meanSquaredError = 0.0;
	vec y = data.getOutputVector();

	for(unsigned int i=0;i<data.getNumberOfSamples(); i++){

		rowvec xp = data.getRowX(i);

		rowvec x  = data.getRowXRaw(i);

#if 0
		printf("\nData point = %d\n", i+1);
		printf("Interpolation at x:\n");
		x.print();
		printf("xnorm:\n");
		xp.print();
#endif
		double functionValueSurrogate = interpolate(xp);

		double functionValueExact = y(i);

		double squaredError = (functionValueExact-functionValueSurrogate)*(functionValueExact-functionValueSurrogate);

		meanSquaredError+= squaredError;
#if 0
		printf("func_val (exact) = %15.10f, func_val (approx) = %15.10f, squared error = %15.10f\n", functionValueExact,functionValueSurrogate,squaredError);
#endif


	}

	meanSquaredError = meanSquaredError/data.getNumberOfSamples();


	return meanSquaredError;


}

void SurrogateModel::calculateOutSampleError(void){

	assert(ifInitialized);
	assert(ifHasTestData);
	unsigned int numberOfEntries = data.getDimension() + 3;

	testResults = zeros<mat>(NTest, numberOfEntries);




	for(unsigned int i=0;i<NTest;i++){

		rowvec xp = XTest.row(i);

		rowvec x  = XTestraw.row(i);


		double functionValueSurrogate = interpolate(xp);

		double functionValueExact = yTest(i);

		double squaredError = (functionValueExact-functionValueSurrogate)*(functionValueExact-functionValueSurrogate);


		rowvec sample(numberOfEntries);
		copyRowVector(sample,x);
		sample(data.getDimension()) =   functionValueExact;
		sample(data.getDimension()+1) = functionValueSurrogate;
		sample(data.getDimension()+2) = squaredError;

		testResults.row(i) = sample;
	}



}


double SurrogateModel::getOutSampleErrorMSE(void) const{

	assert(testResults.n_rows > 0);

	vec squaredError = testResults.col(data.getDimension()+2);


	return mean(squaredError);



}



void SurrogateModel::saveTestResults(void) const{

	field<std::string> header(testResults.n_cols);

	for(unsigned int i=0; i<data.getDimension(); i++){

		header(i) ="x"+std::to_string(i+1);

	}


	header(data.getDimension())   = "True value";
	header(data.getDimension()+1) = "Estimated value";
	header(data.getDimension()+2) = "Squared Error";

	testResults.save( csv_name(filenameTestResults, header) );


}

void SurrogateModel::visualizeTestResults(void) const{

	std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_Test_Results.py "+ name;

	executePythonScript(python_command);

}





void SurrogateModel::printSurrogateModel(void) const{

	data.print();
	boxConstraints.print();


}





vec SurrogateModel::getxmin(void) const{


	return boxConstraints.getLowerBounds();

}
vec SurrogateModel::getxmax(void) const{

	return boxConstraints.getUpperBounds();

}


rowvec SurrogateModel::getRowX(unsigned int index) const{

	return data.getRowX(index);

}


rowvec SurrogateModel::getRowXRaw(unsigned int index) const{

	return data.getRowXRaw(index);
}




void SurrogateModel::setNameOfInputFileTest(string filename){

	assert(isNotEmpty(filename));
	filenameDataInputTest = filename;


}

void SurrogateModel::tryOnTestData(void) const{

	output.printMessage("Trying surrogate model on test data...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfEntries = dim + 1;
	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();

	mat results(numberOfTestSamples,numberOfEntries);

	for(unsigned int i=0; i<numberOfTestSamples; i++){

		rowvec xp = data.getRowXTest(i);
		rowvec x  = data.getRowXRawTest(i);

		double fTilde = interpolate(xp);

		rowvec sample(numberOfEntries);
		copyRowVector(sample,x);
		sample(dim) =  fTilde;

		results.row(i) = sample;


	}

	output.printMessage("Saving surrogate test results in the file: surrogateTest.csv");

	saveMatToCVSFile(results,"surrogateTest.csv");

	output.printMessage("Surrogate test results", results);



}

