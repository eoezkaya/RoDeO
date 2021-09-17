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

PartitionData::PartitionData(){

}

PartitionData::PartitionData(std::string name){

	label = name;

}


void PartitionData::fillWithData(mat inputData){

	assert(inputData.n_rows > 0);

	rawData = inputData;

	numberOfSamples = rawData.n_rows;

	if(ifHasGradientData){

		dim = (rawData.n_cols-1)/2;

	}
	else{

		dim = rawData.n_cols-1;

	}



	X = rawData.submat(0,0,numberOfSamples-1, dim-1);

	if(ifHasGradientData){

		gradientData = rawData.submat(0,dim+1,numberOfSamples-1, 2*dim);

	}

	yExact = rawData.col(dim);
	ySurrogate = zeros<vec>(numberOfSamples);
	squaredError = zeros<vec>(numberOfSamples);
}



void PartitionData::saveAsCSVFile(std::string fileName){

	mat saveBuffer = rawData;
	int nCols = rawData.n_cols;
	saveBuffer.reshape(numberOfSamples,nCols+2);
	saveBuffer.col(nCols) =  ySurrogate;
	saveBuffer.col(nCols+1) =  squaredError;

	saveBuffer.save(fileName,csv_ascii);

}




rowvec PartitionData::getRow(unsigned int indx) const{

	return X.row(indx);

}

double PartitionData::calculateMeanSquaredError(void) const{

	return mean(squaredError);

}

void PartitionData::normalizeAndScaleData(vec xmin, vec xmax){

	X =  normalizeMatrix(X, xmin, xmax);
	X = X*(1.0/dim);
	ifNormalized = true;

}

void PartitionData::print(void) const{

	cout <<"Data: "<<label <<"\n";
	printMatrix(rawData,"rawData");
	printMatrix(X,"X");


}


SurrogateModel::SurrogateModel(){


}

SurrogateModel::SurrogateModel(std::string name){

	assert(isNotEmpty(name));

	label = name;
	filenameDataInput = name +".csv";
	filenameTestResults = name + "_TestResults.csv";

}

void SurrogateModel::checkIfParameterBoundsAreOk(void) const{

	assert(boxConstraints.checkIfBoundsAreValid());


}

void SurrogateModel::setParameterBounds(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());
	boxConstraints = boxConstraintsInput;


}



void SurrogateModel::setParameterBounds(vec xmin, vec xmax){

	Bounds boxConstraints(xmin,xmax);
	setParameterBounds(boxConstraints);

}

void SurrogateModel::setParameterBounds(double xmin, double xmax){

	assert(dim>0);

	Bounds boxConstraints(dim);
	boxConstraints.setBounds(xmin,xmax);

	setParameterBounds(boxConstraints);
}


std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return hyperparameters_filename;

}




std::string SurrogateModel::getNameOfInputFile(void) const{

	return this->filenameDataInput;

}


unsigned int SurrogateModel::getDimension(void) const{

	return this->dim;


}

unsigned int SurrogateModel::getNumberOfSamples(void) const{

	return numberOfSamples;


}

mat SurrogateModel::getRawData(void) const{

	return rawData;


}

void SurrogateModel::checkRawData(void) const{

	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec sample1 = rawData.row(i);

		for(unsigned int j=i+1; j<numberOfSamples; j++){

			rowvec sample2 = rawData.row(j);

			if(checkifTooCLose(sample1, sample2)) {

				printf("ERROR: Two samples in the training data are too close to each other!\n");

				abort();
			}
		}
	}



}

void SurrogateModel::readData(void){

	assert(isNotEmpty(filenameDataInput));

	if(ifDisplay){

		std::cout<<"Loading data from the file "<<filenameDataInput<<"...\n";

	}

	bool status = rawData.load(filenameDataInput.c_str(), csv_ascii);

	if(status == true)
	{
		printMsg("Data input is done");

	}
	else
	{
		std::cout<<"ERROR: Problem with data the input (cvs ascii format)\n";
		std::cout<<"Cannot read: "<<filenameDataInput<<"\n";
		abort();
	}


	if(ifHasGradientData){

		assert((rawData.n_cols - 1)%2 == 0);
		dim = (rawData.n_cols - 1)/2;
	}
	else{

		dim = rawData.n_cols - 1;
	}

	/* set number of sample points */
	numberOfSamples = rawData.n_rows;

	printMsg<unsigned int>("Dimension =",dim);
	printMsg<unsigned int>("Number of samples =", numberOfSamples);


	checkRawData();


	Xraw = rawData.submat(0, 0, numberOfSamples - 1, dim - 1);

	X = Xraw;

	if(ifHasGradientData){

		gradientData = rawData.submat(0, dim+1, numberOfSamples - 1, 2*dim);

	}

	y = rawData.col(dim);


	ymin = min(y);
	ymax = max(y);
	yave = mean(y);

	ifDataIsRead = true;

}


void SurrogateModel::normalizeData(void){

	assert(ifDataIsRead);

	X = normalizeMatrix(X,boxConstraints);

	X = (1.0/dim)*X;
	ifNormalized = true;

}

void SurrogateModel::tryModelOnTestSet(PartitionData &testSet) const{

	/* testset should be without gradients */
	assert(testSet.dim == dim);

	unsigned int howManySamples = testSet.numberOfSamples;


	/* normalize data matrix for the Validation */

	if(testSet.ifNormalized == false){

		vec xmin = this->boxConstraints.getLowerBounds();
		vec xmax = this->boxConstraints.getUpperBounds();

		testSet.normalizeAndScaleData(xmin,xmax);

	}

	for(unsigned int i=0; i<howManySamples; i++){

		rowvec x = testSet.getRow(i);

		testSet.ySurrogate(i) = interpolate(x);


		testSet.squaredError(i) = (testSet.ySurrogate(i)-testSet.yExact(i)) * (testSet.ySurrogate(i)-testSet.yExact(i));
#if 0
		printf("\nx: ");
		x.print();
		printf("fExactValue = %15.10f, fSurrogateValue = %15.10f\n",testSet.yExact(i),testSet.ySurrogate(i));
#endif

	}


}

void SurrogateModel::updateAuxilliaryFields(void){




}




void SurrogateModel::setTestData(mat testData){

	assert(testData.n_rows > 0);
	assert(testData.n_cols == dim +1);
	assert(boxConstraints.areBoundsSet());

	NTest = testData.n_rows;

	XTestraw = testData.submat(0,0,NTest-1, dim-1);

	yTest = testData.col(dim);

	XTest = normalizeMatrix(XTestraw, this->boxConstraints);

	XTest = (1.0/dim)*XTest;

	ifHasTestData = true;
}




double SurrogateModel::calculateInSampleError(void) const{

	assert(ifInitialized);

	double meanSquaredError = 0.0;

	for(unsigned int i=0;i<numberOfSamples;i++){

		rowvec xp = X.row(i);

		rowvec x  = getRowXRaw(i);

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

	meanSquaredError = meanSquaredError/numberOfSamples;


	return meanSquaredError;


}

void SurrogateModel::calculateOutSampleError(void){

	assert(ifInitialized);
	assert(ifHasTestData);
	unsigned int numberOfEntries = dim + 3;

	testResults = zeros<mat>(NTest, numberOfEntries);




	for(unsigned int i=0;i<NTest;i++){

		rowvec xp = XTest.row(i);

		rowvec x  = XTestraw.row(i);

		if(ifPrintOutSampleError){
			std::cout<<"\n"<<i+1<<") Data point at x = \n";

			x.print();
		}
		double functionValueSurrogate = interpolate(xp);

		double functionValueExact = yTest(i);

		double squaredError = (functionValueExact-functionValueSurrogate)*(functionValueExact-functionValueSurrogate);


		if(ifPrintOutSampleError){

			printf("True value = %15.10f, Estimated value = %15.10f, Squared error = %15.10f\n", functionValueExact,functionValueSurrogate,squaredError);
		}


		rowvec sample(numberOfEntries);
		copyRowVector(sample,x);
		sample(dim) =   functionValueExact;
		sample(dim+1) = functionValueSurrogate;
		sample(dim+2) = squaredError;

		testResults.row(i) = sample;
	}



}


double SurrogateModel::getOutSampleErrorMSE(void) const{

	assert(testResults.n_rows > 0);
	vec squaredError = this->testResults.col(dim+2);


	return mean(squaredError);



}



void SurrogateModel::saveTestResults(void) const{

	field<std::string> header(testResults.n_cols);

	for(unsigned int i=0; i<dim; i++){

		header(i) ="x"+std::to_string(i+1);

	}


	header(dim)   = "True value";
	header(dim+1) = "Estimated value";
	header(dim+2) = "Squared Error";

	testResults.save( csv_name(filenameTestResults, header) );


}

void SurrogateModel::visualizeTestResults(void) const{

	std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_Test_Results.py "+ label;

	executePythonScript(python_command);

}





void SurrogateModel::printSurrogateModel(void) const{

	assert(numberOfSamples>0);

	cout << "Surrogate model:"<< endl;
	cout<< "Number of samples: "<<numberOfSamples<<endl;
	cout<<"Number of input parameters: "<<dim<<endl;
	cout<<"Raw Data:\n";
	rawData.print();

	boxConstraints.print();

	cout<<"ymin = "<<ymin<<endl;
	cout<<"ymax = "<<ymax<<endl;
	cout<<"ymean = "<<yave<<endl;
}





vec SurrogateModel::getxmin(void) const{


	return boxConstraints.getLowerBounds();

}
vec SurrogateModel::getxmax(void) const{

	return boxConstraints.getUpperBounds();

}


rowvec SurrogateModel::getRowX(unsigned int index) const{

	assert(index < X.n_rows);

	return X.row(index);

}


rowvec SurrogateModel::getRowXRaw(unsigned int index) const{

	assert(index < X.n_rows);

	return Xraw.row(index);
}


void SurrogateModel::printMsg(std::string message) const{

	assert(isNotEmpty(message));

	if(ifDisplay){

		std::cout<<message<<"\n";

	}


}

template <class T>
void SurrogateModel::printMsg(std::string message, T whatToPrint) const{



	if(ifDisplay){

		std::cout<<message<<" ";
		std::cout<<whatToPrint<<"\n";

	}


}


