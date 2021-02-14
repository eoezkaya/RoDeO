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

	ifNormalized = false;
	ifHasGradientData = false;
	numberOfSamples = 0;
	dim = 0;
}

PartitionData::PartitionData(std::string name){

	label = name;
	ifNormalized = false;
	ifHasGradientData = false;
	numberOfSamples = 0;
	dim = 0;
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

	modelID = NONE;
	dim = 0;
	label = "None";
	N = 0;
	numberOfHyperParameters = 0;
	ifInitialized = false;
	ifUsesGradientData = false;
	ifprintToScreen = false;
}

SurrogateModel::SurrogateModel(std::string name){

	dim = 0;
	N = 0;
	numberOfHyperParameters = 0;
	label = name;
	input_filename = name +".csv";
	ifInitialized = false;
	ifUsesGradientData = false;
	ifprintToScreen = false;

}


void SurrogateModel::ReadDataAndNormalize(void){

#if 0
	std::cout<<"Loading data from the file "<<input_filename<<"...\n";
#endif
	bool status = rawData.load(input_filename.c_str(), csv_ascii);

	if(status == true)
	{
#if 0
		printf("Data input is done\n");
#endif
	}
	else
	{
		printf("ERROR: Problem with data the input (cvs ascii format)\n");
		abort();
	}

	if(ifUsesGradientData){

		assert((rawData.n_cols - 1)%2 == 0);
		dim = (rawData.n_cols - 1)/2;
	}
	else{

		dim = rawData.n_cols - 1;
	}

	/* set number of sample points */
	N = rawData.n_rows;

	X = rawData.submat(0, 0, N - 1, dim - 1);


	if(ifUsesGradientData){

		gradientData = rawData.submat(0, dim+1, N - 1, 2*dim);

	}


	xmax.set_size(dim);
	xmin.set_size(dim);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = rawData.col(i).max();
		xmin(i) = rawData.col(i).min();

	}

	X = normalizeMatrix(X);
	X = (1.0/dim)*X;


	y = rawData.col(dim);


	ymin = min(y);
	ymax = max(y);
	yave = mean(y);

}

void SurrogateModel::updateData(mat dataMatrix){

	rawData.reset();
	X.reset();
	gradientData.reset();
	y.reset();

	unsigned int dimDataMatrix;

	if(ifUsesGradientData){

		dimDataMatrix = (dataMatrix.n_cols - 1)/2;

	}
	else{

		dimDataMatrix = dataMatrix.n_cols - 1;

	}



	if(dimDataMatrix !=dim){

		cout<<"ERROR: Dimension of the new data does not match with the problem dimension!\n";
		cout<<"dimDataMatrix = "<<dimDataMatrix<<"\n";
		cout<<"dim = "<<dim<<"\n";
		abort();


	}

	rawData = dataMatrix;

	/* set number of sample points */
	N = rawData.n_rows;

	X = rawData.submat(0, 0, N - 1, dim - 1);

	if(ifUsesGradientData){

		gradientData = rawData.submat(0, dim+1, N - 1, 2*dim);

	}


	xmax.set_size(dim);
	xmin.set_size(dim);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = rawData.col(i).max();
		xmin(i) = rawData.col(i).min();

	}

	X = normalizeMatrix(X);
	X = (1.0/dim)*X;

	y = rawData.col(dim);


	ymin = min(y);
	ymax = max(y);
	yave = mean(y);


}



void SurrogateModel::tryModelOnTestSet(PartitionData &testSet) const{

	/* testset should be without gradients */
	assert(testSet.dim == dim);

	unsigned int howManySamples = testSet.numberOfSamples;


	/* normalize data matrix for the Validation */

	if(testSet.ifNormalized == false){

		testSet.normalizeAndScaleData(xmin,xmax);

	}

	for(unsigned int i=0; i<howManySamples; i++){

		rowvec x = testSet.getRow(i);

		if(ifUsesGradientData){

			testSet.ySurrogate(i) = interpolateWithGradients(x);
		}
		else{

			testSet.ySurrogate(i) = interpolate(x);
		}

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

double SurrogateModel::calculateInSampleError(void) const{


	double meanSquaredError = 0.0;

	for(unsigned int i=0;i<N;i++){

		rowvec xp = X.row(i);

		rowvec x  = getRowXRaw(i);

#if 1
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
#if 1
		printf("func_val (exact) = %15.10f, func_val (approx) = %15.10f, squared error = %15.10f\n", functionValueExact,functionValueSurrogate,squaredError);
#endif


	}

	meanSquaredError = meanSquaredError/N;


	return meanSquaredError;


}


void SurrogateModel::visualizeTestResults(void) const{

	std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_Test_Results.py "+ label;

	executePythonScript(python_command);

}





void SurrogateModel::printSurrogateModel(void) const{

	assert(N>0);

	cout << "Surrogate model:"<< endl;
	cout<< "Number of samples: "<<N<<endl;
	cout<<"Number of input parameters: "<<dim<<endl;
	cout<<"Raw Data:\n";
	rawData.print();
	cout<<"xmin =";
	trans(xmin).print();
	cout<<"xmax =";
	trans(xmax).print();
	cout<<"ymin = "<<ymin<<endl;
	cout<<"ymax = "<<ymax<<endl;
	cout<<"ymean = "<<yave<<endl;
}


std::string SurrogateModel::getInputFileName(void) const{

	return input_filename;

}


/** Returns the ith row of the matrix X
 * @param[in] index
 *
 */
rowvec SurrogateModel::getRowX(unsigned int index) const{

	assert(index < X.n_rows);

	return X.row(index);

}

/** Returns the ith row of the matrix X in raw data format
 * @param[in] index
 *
 */
rowvec SurrogateModel::getRowXRaw(unsigned int index) const{

	assert(index < X.n_rows);

	rowvec x(dim);
	rowvec xnorm = X.row(index);

	for(unsigned int i=0; i<dim; i++){

		x(i) = xnorm(i)*dim * (xmax(i) - xmin(i)) + xmin(i);
	}

	return x;

}



