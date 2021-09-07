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
#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;


LinearModel::LinearModel(std::string name):SurrogateModel(name){

	modelID = LINEAR_REGRESSION;
	regularizationParam = 10E-6;
	hyperparameters_filename = label + "_linear_regression_hyperparameters.csv";


}

LinearModel::LinearModel():SurrogateModel(){


}


void LinearModel::initializeSurrogateModel(void){

	numberOfHyperParameters = dim+1;
	weights = zeros<vec>(numberOfHyperParameters);

	ifInitialized = true;

}

void LinearModel::saveHyperParameters(void) const  {

	weights.save(hyperparameters_filename, csv_ascii);

}

void LinearModel::loadHyperParameters(void){

	weights.load(hyperparameters_filename, csv_ascii);

}

void LinearModel::printHyperParameters(void) const{

	printVector(weights,"linear regression weights");

}





void LinearModel::setRegularizationParam(double value){

	regularizationParam = value;
}

double LinearModel::getRegularizationParam(void) const{

	return regularizationParam;
}

vec LinearModel::getWeights(void) const{

	return weights;
}

void LinearModel::setWeights(vec w){

	weights = w;
}



void LinearModel::train(void){

	if(ifInitialized == false){

		printf("ERROR: Linear regression model must be initialized before training!\n");
		abort();
	}

	mat augmented_X(N, dim + 1);

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j <= dim; j++) {

			if (j == 0){

				augmented_X(i, j) = 1.0;
			}

			else{

				augmented_X(i, j) = X(i, j - 1);
			}


		}
	}

#if 0
	printf("augmented_X:\n");
	augmented_X.print();
#endif



	if(fabs(regularizationParam) < EPSILON ){
#if 0
		printf("Taking pseudo-inverse of augmented data matrix...\n");
#endif
		mat psuedo_inverse_X_augmented = pinv(augmented_X);

		//		psuedo_inverse_X_augmented.print();

		weights = psuedo_inverse_X_augmented * y;

	}

	else{
#if 0
		printf("Regularization...\n");
#endif
		mat XtX = trans(augmented_X)*augmented_X;

		XtX = XtX + regularizationParam*eye(XtX.n_rows,XtX.n_rows);

		weights = inv(XtX)*trans(augmented_X)*y;

	}

	for(unsigned int i=0; i<dim+1;i++ ){

		if(fabs(weights(i)) > 10E5){

			printf("WARNING: Linear regression coefficients are too large= \n");
			printf("regression_weights(%d) = %10.7f\n",i,weights(i));
		}

	}

	if(ifprintToScreen){

		printf("Linear regression weights:\n");
		trans(weights).print();

	}

}




double LinearModel::interpolate(rowvec x ) const{

	double fRegression = 0.0;
	for(unsigned int i=0; i<dim; i++){

		fRegression += x(i)*weights(i+1);
	}

	/* add bias term */
	fRegression += weights(0);

	return fRegression;


}



void LinearModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{

	cout << "ERROR: interpolateWithVariance does not exist for LinearModel\n";
	abort();


}
double LinearModel::interpolateWithGradients(rowvec xp) const{

	cout << "ERROR: interpolateWithGradients does not exist for LinearModel\n";
	abort();


}

vec LinearModel::interpolateAll(mat X) const{

	unsigned int numberOfSamples = X.n_rows;

	vec result(numberOfSamples);


	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec x = X.row(i);

		double fRegression = 0.0;
		for(unsigned int j=0; j<dim; j++){

			fRegression += x(j)*weights(j+1);
		}

		/* add bias term */
		fRegression += weights(0);

		result(i) = fRegression;
	}

	return result;


}




void LinearModel::printSurrogateModel(void) const{

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
	cout<<"Regression weights:\n";
	trans(weights).print();

}

void LinearModel::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{

	std::cout<<"ERROR: Cannot calculate expected improvement for linear model!\n";
	abort();


}

void LinearModel::addNewSampleToData(rowvec newsample){




}


void LinearModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;
	ifInputFilenameIsSet = true;

}



void LinearModel::setNumberOfTrainingIterations(unsigned int nIters){}

void LinearModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());
	hyperparameters_filename = filename;

}


