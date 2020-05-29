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


SurrogateModel::SurrogateModel(){

	modelID = NONE;
	dim = 0;
	label = "None";
	N = 0;


}




SurrogateModel::SurrogateModel(std::string name, unsigned int dimension){

	dim = dimension;

	label = name;
	printf("Initiating surrogate model for the %s data (Base model)...\n",name.c_str());
	input_filename = name +".csv";
	hyperparameters_filename = name + "_Hyperparameters.csv";


#if 1
	std::cout<<"Loading data from the file "<<input_filename<<"...\n";
#endif
	bool status = data.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Error: Problem with data the input (cvs ascii format) at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}



	assert(data.n_cols == dim+1);

	/* set number of sample points */
	N = data.n_rows;

	X = data.submat(0, 0, N - 1, dim - 1);


	xmax = zeros(dim);
	xmin = zeros(dim);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = data.col(i).max();
		xmin(i) = data.col(i).min();

	}

	normalizeInputMatrix();



	ymin = min(data.col(dim));
	ymax = max(data.col(dim));
	yave = mean(data.col(dim));

}

void SurrogateModel::normalizeInputMatrix(void){

	for (unsigned int i = 0; i < N; i++) {

			for (unsigned int j = 0; j < dim; j++) {

				X(i, j) = (1.0/dim)*(X(i, j) - xmin(j)) / (xmax(j) - xmin(j));
			}
		}

}



mat SurrogateModel::tryModelOnTestSet(mat testSet) const{

	/* testset should be without gradients */
	assert(testSet.n_cols == dim+1);

	unsigned int howManySamples = testSet.n_rows;

	vec fSurrogateValue(howManySamples);
	vec squaredError(howManySamples);
	vec fExactValue = testSet.col(dim);

	mat xTest = testSet.submat(0, 0, howManySamples - 1, dim - 1);

	/* normalize data matrix for the Validation */

	for (unsigned int i = 0; i < howManySamples; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			xTest(i, j) = (1.0/dim)*(xTest(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}
	}


	for(unsigned int i=0; i<howManySamples; i++){

		fSurrogateValue(i) = interpolate(xTest.row(i));
		squaredError(i) = (fSurrogateValue(i)-fExactValue(i)) * (fSurrogateValue(i)-fExactValue(i));
#if 1
		printf("\nx: ");
		xTest.row(i).print();
		printf("fExactValue = %15.10f, fExactValue = %15.10f\n",fSurrogateValue(i),fExactValue(i));
#endif

	}

	mat testResults = testSet;
	testResults.reshape(howManySamples,dim+3);
	testResults.col(dim+1) =  fSurrogateValue;
	testResults.col(dim+2) =  squaredError;

	testResults.print();




	return testResults;
}

void SurrogateModel::visualizeTestResults(mat testResults) const{


	std::string python_command;

	python_command = "python -W ignore "+ settings.python_dir + "/plot_Test_Results.py "+ label;

	executePythonScript(python_command);


}

void SurrogateModel::train(void){

	fprintf(stderr, "ERROR: cannot train the base class: SurrogateModel! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);

}

double SurrogateModel::interpolate(rowvec x) const{

	fprintf(stderr, "ERROR: cannot interpolate using the base class: SurrogateModel! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);

}

void SurrogateModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{

	fprintf(stderr, "ERROR: cannot interpolate using the base class: SurrogateModel! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);

}


double SurrogateModel::calculateInSampleError(void) const{

	fprintf(stderr, "ERROR: cannot call  calculateInSampleError using the base class: SurrogateModel! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);

}

void SurrogateModel::print(void) const{

	cout << "Surrogate model:"<< endl;
	cout<< "Number of samples: "<<N<<endl;
	cout<<"Number of input parameters: "<<dim<<endl;
	cout<<"Raw Data:\n";
	data.print();
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



