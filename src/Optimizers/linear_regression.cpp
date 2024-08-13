/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#include "linear_regression.hpp"
#include "surrogate_model.hpp"
#include "auxiliary_functions.hpp"
#include "bounds.hpp"
#include "Rodeo_macros.hpp"


#include <armadillo>

using namespace arma;



//LinearModel::LinearModel():SurrogateModel(){}

void LinearModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
}


void LinearModel::readData(void){

	assert(isNotEmpty(filenameDataInput));
	data.readData(filenameDataInput);
	numberOfSamples = data.getNumberOfSamples();

	ifDataIsRead = true;

}

void LinearModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalize();
	ifNormalized = true;
}


void LinearModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(dimension >0);

	output.printMessage("Initializing the linear model...");

	numberOfHyperParameters = dimension+1;
	weights = zeros<vec>(numberOfHyperParameters);
	regularizationParameter  = 10E-6;

	train();

	ifInitialized = true;

}

void LinearModel::saveHyperParameters(void) const  {

	assert(isNotEmpty(filenameHyperparameters));
	weights.save(filenameHyperparameters, csv_ascii);

}

void LinearModel::loadHyperParameters(void){

	assert(isNotEmpty(filenameHyperparameters));
	weights.load(filenameHyperparameters, csv_ascii);

}

void LinearModel::printHyperParameters(void) const{

	weights.print("linear regression weights");

}

void LinearModel::setRegularizationParameter(double value){

	regularizationParameter = value;
}

double LinearModel::getRegularizationParameter(void) const{
	return regularizationParameter;
}
vec LinearModel::getWeights(void) const{
	return weights;
}

void LinearModel::setWeights(vec w){
	weights = w;
}

void LinearModel::train(void){

	assert(ifNormalized);
	assert(ifDataIsRead);
	assert(dimension>0);
	assert(numberOfSamples>0);

	output.printMessage("Finding the weights of the linear model...");


	mat X = data.getInputMatrix();
	vec y = data.getOutputVector();
	mat augmentedX(numberOfSamples, dimension + 1);

	for (unsigned int i = 0; i < numberOfSamples; i++) {

		for (unsigned int j = 0; j <= dimension; j++) {

			if (j == 0){

				augmentedX(i, j) = 1.0;
			}

			else{

				augmentedX(i, j) = X(i, j - 1);
			}
		}
	}

	mat XT = trans(augmentedX);
	mat XTX = XT *augmentedX;
	XTX = XTX + regularizationParameter * eye(dimension+1,dimension+1);
	weights = inv(XTX)*XT*y;

	output.printMessage("Linear regression weights", weights);

	ifModelTrainingIsDone = true;


}

double LinearModel::interpolate(rowvec x ) const{

	double fRegression = weights(0);

	for(unsigned int i=0; i<dimension; i++){

		fRegression += x(i)*weights(i+1);
	}

	return fRegression;

}

double LinearModel::interpolateUsingDerivatives(rowvec x ) const{

	abortWithErrorMessage("Linear model cannot interpolate using derivatives!");
}


void LinearModel::interpolateWithVariance(rowvec xp,double *fTilde,double *ssqr) const{

	assert(false);
	xp.print();
	*fTilde = 0.0;
	*ssqr = 0.0;

}

vec LinearModel::interpolateAll(mat X) const{

	assert(X.n_cols == dimension);

	unsigned int N = X.n_rows;

	vec result(N,fill::zeros);

	for(unsigned int i=0; i<N; i++){
		result(i) = interpolate(X.row(i));
	}

	return result;
}


void LinearModel::printSurrogateModel(void) const{

	data.print();
	cout<<"Regression weights:\n";
	trans(weights).print();

}


void LinearModel::addNewSampleToData(rowvec newsample){

	cout<<newsample.size();
	assert(false);
}

void LinearModel::addNewLowFidelitySampleToData(rowvec newsample){
	cout<<newsample.size();
	assert(false);
}


void LinearModel::setNameOfInputFile(std::string filename){

	assert(isNotEmpty(filename));
	filenameDataInput = filename;

}

void LinearModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}


void LinearModel::setNameOfHyperParametersFile(std::string filename){

	assert(isNotEmpty(filename));
	filenameHyperparameters = filename;

}

void LinearModel::updateModelWithNewData(void){



}

