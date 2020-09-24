#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "surrogate_model.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"



#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

PartitionData::PartitionData(){

	ifNormalized = false;
	numberOfSamples = 0;
	dim = 0;
}

PartitionData::PartitionData(std::string name){

	label = name;
	ifNormalized = false;
	numberOfSamples = 0;
	dim = 0;
}


void PartitionData::fillWithData(mat inputData){

	assert(inputData.n_rows > 0);
	assert(inputData.n_cols >=2);

	rawData = inputData;

	numberOfSamples = rawData.n_rows;
	dim = rawData.n_cols-1;
	X = rawData.submat(0,0,numberOfSamples-1, dim-1);

	yExact = rawData.col(dim);
	ySurrogate = zeros<vec>(numberOfSamples);
	squaredError = zeros<vec>(numberOfSamples);
}

void PartitionData::fillWithData(mat inputData, mat gradients){

	assert(inputData.n_rows == gradients.n_rows);
	assert(inputData.n_cols-1 == gradients.n_cols);
	fillWithData(inputData);
	gradientData = gradients;


}


void PartitionData::saveAsCSVFile(std::string fileName){

	mat saveBuffer = rawData;
	saveBuffer.reshape(numberOfSamples,dim+3);
	saveBuffer.col(dim+1) =  ySurrogate;
	saveBuffer.col(dim+2) =  squaredError;

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
	ifInitialized = false;
	ifUsesGradientData = false;

}

SurrogateModel::SurrogateModel(std::string name){

	dim = 0;
	N = 0;

	label = name;
	input_filename = name +".csv";
	ifInitialized = false;
	ifUsesGradientData = false;

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



double SurrogateModel::calculateInSampleError(void) const{


	double meanSquaredError = 0.0;

	for(unsigned int i=0;i<N;i++){

		rowvec xp = getRowX(i);

		rowvec x  = getRowXRaw(i);

#if 1
		printf("\nData point = %d\n", i+1);
		printf("calling f_tilde at x:\n");
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



