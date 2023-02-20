/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), RPTU
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



#include "surrogate_model_data.hpp"
#include "auxiliary_functions.hpp"




SurrogateModelData::SurrogateModelData(){}


void SurrogateModelData::reset(void){

	X.reset();
	XTest.reset();
	Xraw.reset();
	XrawTest.reset();
	dimension = 0;
	gradient.reset();
	differentiationDirections.reset();
	directionalDerivatives.reset();
	ifDataHasGradients = false;
	ifDataIsNormalized = false;
	ifDataIsRead = false;
	numberOfSamples = 0;
	numberOfTestSamples = 0;
	boxConstraints.reset();

}

unsigned int SurrogateModelData::getNumberOfSamples(void) const{


	return numberOfSamples;
}

unsigned int SurrogateModelData::getNumberOfSamplesTest(void) const{


	return numberOfTestSamples;
}


unsigned int SurrogateModelData::getDimension(void) const{


	return dimension;
}


void SurrogateModelData::setDimension(unsigned int value){

	dimension = value;

}


bool SurrogateModelData::isDataRead(void) const{

	return ifDataIsRead;

}

mat SurrogateModelData::getRawData(void) const{

	return rawData;


}


void SurrogateModelData::setDisplayOn(void){

	outputToScreen.ifScreenDisplay = true;
	outputToScreen.printMessage("Setting display on for the SurrogateModelData...");

}

void SurrogateModelData::setDisplayOff(void){

	outputToScreen.ifScreenDisplay = false;


}

void SurrogateModelData::setGradientsOn(void){

	ifDataHasGradients = true;

}

void SurrogateModelData::setGradientsOff(void){

	ifDataHasGradients = false;

}

void SurrogateModelData::setDirectionalDerivativesOn(void){

	ifDataHasDirectionalDerivatives = true;

}

void SurrogateModelData::setDirectionalDerivativesOff(void){

	ifDataHasDirectionalDerivatives = false;

}


void SurrogateModelData::readData(string inputFilename){

	assert(isNotEmpty(inputFilename));
	assert(!(ifDataHasDirectionalDerivatives == true && this->ifDataHasGradients == true));

	outputToScreen.printMessage("Loading data from the file: " + inputFilename);

	bool status = rawData.load(inputFilename.c_str(), csv_ascii);

	if(status == true)
	{
		outputToScreen.printMessage("Data input is done...");

	}
	else
	{
		outputToScreen.printErrorMessageAndAbort("Problem with data the input (cvs ascii format), cannot read: " + inputFilename);

	}

	numberOfSamples = rawData.n_rows;
	outputToScreen.printMessage("Number of samples = ", numberOfSamples);
	outputToScreen.printMessage("Raw data = ", rawData);



	assignDimensionFromData();

	assignSampleInputMatrix();
	assignSampleOutputVector();

	assignGradientMatrix();

	assignDirectionalDerivativesVector();
	assignDifferentiationDirectionMatrix();

	ifDataIsRead = true;


}

void SurrogateModelData::readDataTest(string inputFilename){

	assert(isNotEmpty(inputFilename));
	assert(dimension>0);

	outputToScreen.printMessage("Loading test data from the file: " + inputFilename);

	bool status = XrawTest.load(inputFilename.c_str(), csv_ascii);
	if(status == true)
	{
		outputToScreen.printMessage("Data input is done...");
	}
	else
	{
		outputToScreen.printErrorMessageAndAbort("Problem with data the input (cvs ascii format), cannot read: " + inputFilename);
	}

	numberOfTestSamples = XrawTest.n_rows;
	outputToScreen.printMessage("Number of test samples = ", numberOfTestSamples);

	if(XrawTest.n_cols == dimension +1){

		yTest = XrawTest.col(dimension);
		XTest = XrawTest.submat(0,0,numberOfTestSamples-1, dimension-1);
		ifTestDataHasFunctionValues = true;
	}

	if(XrawTest.n_cols == dimension){
		XTest = XrawTest;
	}
	if(XrawTest.n_cols < dimension || XrawTest.n_cols> dimension+1){
		outputToScreen.printErrorMessageAndAbort("Problem with data the test data (cvs ascii format), too many or too few columns in file: " + inputFilename);
	}

	ifTestDataIsRead = true;
}



void SurrogateModelData::assignDimensionFromData(void){

	unsigned int dimensionOfTrainingData;

	if(ifDataHasGradients){
		dimensionOfTrainingData = (rawData.n_cols-1)/2;
	}
	else if(ifDataHasDirectionalDerivatives){
		dimensionOfTrainingData =  (rawData.n_cols-2)/2;
	}
	else{
		dimensionOfTrainingData =  rawData.n_cols-1;
	}

	if(dimension > 0 && dimensionOfTrainingData!= dimension){
		outputToScreen.printErrorMessageAndAbort("Dimension of the training data does not match with the specified dimension!");
	}

	dimension = dimensionOfTrainingData;
	outputToScreen.printMessage("Dimension of the problem is identified as ", dimension);

}

void SurrogateModelData::assignSampleInputMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	X = rawData.submat(0,0,numberOfSamples-1, dimension-1);
	Xraw = X;

}

void SurrogateModelData::assignSampleOutputVector(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	y = rawData.col(dimension);

}

void SurrogateModelData::assignGradientMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);


	if(ifDataHasGradients){

		assert(rawData.n_cols > dimension+1);

		gradient = rawData.submat(0, dimension+1, numberOfSamples - 1, 2*dimension);

	}


}


void SurrogateModelData::assignDirectionalDerivativesVector(void){

	assert(dimension>0);

	if(ifDataHasDirectionalDerivatives){
		directionalDerivatives = rawData.col(dimension+1);
	}
}


void SurrogateModelData::assignDifferentiationDirectionMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	if(ifDataHasDirectionalDerivatives){
		differentiationDirections = rawData.submat(0, dimension+2,numberOfSamples-1,  2*dimension+1);
	}
}

void SurrogateModelData::normalize(void){

	assert(ifDataIsRead);
	assert(boxConstraints.areBoundsSet());

	normalizeSampleInputMatrix();
	normalizeDerivativesMatrix();

	ifDataIsNormalized = true;
}


void SurrogateModelData::normalizeSampleInputMatrix(void){

	assert(X.n_rows ==  numberOfSamples);
	assert(X.n_cols ==  dimension);
	assert(boxConstraints.getDimension() == dimension);
	assert(boxConstraints.areBoundsSet());

	outputToScreen.printMessage("Normalizing and scaling the sample input matrix...");

	mat XNormalized = X;
	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;

	for(unsigned int i=0; i<numberOfSamples;i++){
		for(unsigned int j=0; j<dimension;j++){

			XNormalized(i,j) = (X(i,j) - xmin(j))/deltax(j);
		}
	}

	X = (1.0/dimension)*XNormalized;

	ifDataIsNormalized = true;
}


void SurrogateModelData::normalizeSampleInputMatrixTest(void){

	assert(boxConstraints.areBoundsSet());


	outputToScreen.printMessage("Normalizing and scaling the sample input matrix for test...");

	mat XNormalized = XTest;
	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;


	for(unsigned int i=0; i<numberOfTestSamples;i++){
		for(unsigned int j=0; j<dimension;j++){
			XNormalized(i,j) = (XTest(i,j) - xmin(j))/deltax(j);
		}
	}

	XTest = (1.0/dimension)*XNormalized;


}

void SurrogateModelData::normalizeDerivativesMatrix(void){

	if(ifDataHasDirectionalDerivatives){

		assert(ifDataIsRead);
		assert(boxConstraints.areBoundsSet());

		vec lb = boxConstraints.getLowerBounds();
		vec ub = boxConstraints.getUpperBounds();
		double scalingFactor = (ub(0) - lb(0))*dimension;
		assert(scalingFactor > 0.0);
		directionalDerivatives = scalingFactor * directionalDerivatives;


	}

}


void SurrogateModelData::normalizeGradientMatrix(void){

	if(ifDataHasGradients){


		assert(ifDataIsRead);
		assert(boxConstraints.areBoundsSet());

		for(unsigned int i=0; i<dimension; i++){

			vec lb = boxConstraints.getLowerBounds();
			vec ub = boxConstraints.getUpperBounds();
			double scalingFactor = ub(i) - lb(i);
			assert(scalingFactor > 0.0);
			gradient.row(i) = scalingFactor * gradient.row(i);



		}


	}

}



void SurrogateModelData::normalizeOutputVector(void){
	assert(ifDataIsRead);
	assert(numberOfSamples>0);
	vec y = getOutputVector();

	double absMax = 0.0;

	for(unsigned int i=0; i<numberOfSamples; i++){

		if(fabs(y(i)) > absMax ) absMax = fabs(y(i));

	}

	scalingFactorOutput = 1.0/absMax;

	y = y*scalingFactorOutput;
	setOutputVector(y);

}


double SurrogateModelData::getScalingFactorForOutput(void) const{

	return scalingFactorOutput;
}



rowvec SurrogateModelData::getRowGradient(unsigned int index) const{

	return gradient.row(index);

}

vec SurrogateModelData::getDirectionalDerivativesVector(void) const{

	return directionalDerivatives;


}


rowvec SurrogateModelData::getRowDifferentiationDirection(unsigned int index) const{

	return differentiationDirections.row(index);

}


rowvec SurrogateModelData::getRowRawData(unsigned int index) const{

	return rawData.row(index);

}



rowvec SurrogateModelData::getRowX(unsigned int index) const{

	assert(index < X.n_rows);

	return X.row(index);

}

rowvec SurrogateModelData::getRowXTest(unsigned int index) const{

	assert(index < XTest.n_rows);

	return XTest.row(index);

}




rowvec SurrogateModelData::getRowXRaw(unsigned int index) const{

	assert(index < Xraw.n_rows);

	return Xraw.row(index);
}

rowvec SurrogateModelData::getRowXRawTest(unsigned int index) const{

	assert(index < XrawTest.n_rows);

	return XrawTest.row(index);
}


vec SurrogateModelData::getOutputVector(void) const{

	return y;

}

vec SurrogateModelData::getOutputVectorTest(void) const{

	assert(ifTestDataIsRead);
	return XrawTest.col(dimension);

}


void SurrogateModelData::setOutputVector(vec yIn){

	assert(yIn.size() == numberOfSamples);
	y = yIn;


}


mat SurrogateModelData::getInputMatrix(void) const{

	return X;

}


double SurrogateModelData::getMinimumOutputVector(void) const{

	return min(y);
}

double SurrogateModelData::getMaximumOutputVector(void) const{

	return max(y);
}

mat SurrogateModelData::getGradientMatrix(void) const{

	return gradient;

}






void SurrogateModelData::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;

}

void SurrogateModelData::setBoxConstraintsFromData(void){


	outputToScreen.printMessage("setting box constraints from the training data...");

	vec maxofX(dimension);
	vec minofX(dimension);



	for(unsigned int i=0; i<dimension; i++){

		minofX(i) = min(Xraw.col(i));
		maxofX(i) = max(Xraw.col(i));


	}



	boxConstraints.setBounds(minofX,maxofX);

	outputToScreen.printMessage("lower bounds", minofX);
	outputToScreen.printMessage("upper bounds", maxofX);



}


Bounds SurrogateModelData::getBoxConstraints(void) const{

	return boxConstraints;

}



bool SurrogateModelData::isDataNormalized(void) const{

	return ifDataIsNormalized;

}


void SurrogateModelData::print(void) const{

	printMatrix(rawData,"raw data");
	printMatrix(X,"sample input matrix");
	printVector(y,"sample output vector");

	printScalar(scalingFactorOutput);

	if(ifDataHasGradients){

		printMatrix(gradient,"sample gradient matrix");

	}

	if(ifDataHasDirectionalDerivatives){

		printVector(directionalDerivatives, "directional derivatives");
		printMatrix(differentiationDirections, "differentiation directions");

	}

	if(ifTestDataIsRead){

		printMatrix(XrawTest,"raw data for testing");

		printMatrix(XTest, "sample input matrix for testing");

	}

}








