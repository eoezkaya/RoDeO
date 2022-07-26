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


void SurrogateModel::setNumberOfThreads(unsigned int n){

	numberOfThreads = n;

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

void SurrogateModel::setWriteWarmStartFileOn(std::string filename){

	assert(isNotEmpty(filename));

	filenameForWriteWarmStart = filename;
	ifWriteWarmStartFile = true;

}


void SurrogateModel::setReadWarmStartFileOn(std::string filename){

	assert(isNotEmpty(filename));
	filenameForWarmStartModelTraining = filename;
	ifReadWarmStartFile = true;

}

void SurrogateModel::setDisplayOn(void){

	data.setDisplayOn();
	output.ifScreenDisplay = true;

}

void SurrogateModel::setDisplayOff(void){

	data.setDisplayOff();
	output.ifScreenDisplay = false;

}


void SurrogateModel::setMinimizeOn(void){

	ifMinimize = true;
	ifMaximize = false;

}

void SurrogateModel::setMaximizeOn(void){

	ifMinimize = false;
	ifMaximize = true;

}



void SurrogateModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	data.setBoxConstraints(boxConstraintsInput);



}



void SurrogateModel::setBoxConstraints(vec xmin, vec xmax){

	Bounds boxConstraints(xmin,xmax);
	setBoxConstraints(boxConstraints);

}

void SurrogateModel::setBoxConstraints(double xmin, double xmax){

	Bounds boxConstraints(data.getDimension());
	boxConstraints.setBounds(xmin,xmax);

	setBoxConstraints(boxConstraints);
}


void SurrogateModel::setBoxConstraintsFromData(void){


	data.setBoxConstraintsFromData();


}

std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return hyperparameters_filename;

}




std::string SurrogateModel::getNameOfInputFile(void) const{

	return filenameDataInput;

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

mat SurrogateModel::getX(void) const{

	return data.getInputMatrix();
}

vec SurrogateModel::gety(void) const{

	return data.getOutputVector();

}

void SurrogateModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.readData(filenameDataInput);

	ifDataIsRead = true;

}

void SurrogateModel::printData(void) const{

	data.print();

}

void SurrogateModel::readDataTest(void){

	assert(isNotEmpty(filenameDataInputTest));

	data.readDataTest(filenameDataInputTest);
	ifHasTestData = true;
}



void SurrogateModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalizeSampleInputMatrix();
	ifNormalized = true;
}

void SurrogateModel::normalizeDataTest(void){

	data.normalizeSampleInputMatrixTest();
	ifNormalizedTestData = true;
}


void SurrogateModel::updateAuxilliaryFields(void){

}


vec SurrogateModel::interpolateVector(mat X) const{

	assert(X.max() <= 1.0/data.getDimension());
	assert(X.min() >= 0.0);

	unsigned int N = X.n_rows;
	vec results(N);

	for(unsigned int i=0; i<N; i++){

		rowvec xp = X.row(i);
		results(i) = interpolate(xp);

	}

	return results;
}


double SurrogateModel::calculateInSampleError(void) const{

	assert(ifInitialized);


	vec fTildeValues = interpolateVector(data.getInputMatrix());
	vec fExact = data.getOutputVector();
	vec diff = fTildeValues - fExact;

	double L2normDiff = norm(diff,2);
	double squaredError = L2normDiff*L2normDiff;

	return squaredError/data.getNumberOfSamples();

}



double SurrogateModel::calculateOutSampleError(void){

	assert(ifHasTestData);
	assert(data.ifTestDataHasFunctionValues);

	tryOnTestData();

	vec squaredErrors = testResults.col(data.getDimension()+2);

	return mean(squaredErrors);

}

void SurrogateModel::saveTestResults(void) const{

	assert(isNotEmpty(filenameTestResults));
	field<std::string> header(testResults.n_cols);

	unsigned int dim = data.getDimension();

	for(unsigned int i=0; i<dim; i++){

		header(i) ="x"+std::to_string(i+1);

	}

	header(dim)   = "Estimated value";
	header(dim+1) = "True value";
	header(dim+2) = "Squared Error";

	testResults.save( csv_name(filenameTestResults, header) );


}


void SurrogateModel::printSurrogateModel(void) const{

	data.print();

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

void SurrogateModel::setNameOfOutputFileTest(string filename){

	assert(isNotEmpty(filename));
	filenameTestResults = filename;

}


void SurrogateModel::tryOnTestData(void){

	assert(ifNormalizedTestData);

	output.printMessage("Trying surrogate model on test data...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfEntries;

	if(data.ifTestDataHasFunctionValues){

		numberOfEntries = dim + 3;
	}
	else{

		numberOfEntries = dim + 1;
	}

	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();

	mat results(numberOfTestSamples,numberOfEntries);

	vec fExact;
	if(data.ifTestDataHasFunctionValues){

		fExact = data.getOutputVectorTest();
	}


	for(unsigned int i=0; i<numberOfTestSamples; i++){

		rowvec xp = data.getRowXTest(i);
		rowvec x  = data.getRowXRawTest(i);

		output.printMessage("x",x);

		double fTilde = interpolate(xp);
		output.printMessage("fTilde = ",fTilde);

		rowvec sample = x.head(dim);
		addOneElement(sample,fTilde);

		if(data.ifTestDataHasFunctionValues){

			addOneElement(sample,fExact(i));
			double squaredError = pow((fExact(i) - fTilde),2.0);
			addOneElement(sample,squaredError);

			output.printMessage("fExact = ",fExact(i));
		}


		results.row(i) = sample;


	}

	output.printMessage("Saving surrogate test results in the file: surrogateTest.csv");

	saveMatToCVSFile(results,  "surrogateTest.csv");

	output.printMessage("Surrogate test results", results);

	testResults = results;



}

