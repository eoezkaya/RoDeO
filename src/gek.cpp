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

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"

#include "Rodeo_globals.hpp"
#include "Rodeo_macros.hpp"
#include "gek.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals_GEK;

double population_overall_max_GEK = -10E14;
int population_overall_max_tread_id_GEK = -1;

GEKModel::GEKModel():SurrogateModel(){}


GEKModel::GEKModel(std::string nameInput):SurrogateModel(nameInput){

	modelID = GRADIENT_ENHANCED_KRIGING;
	setName(nameInput);
	setNameOfHyperParametersFile(nameInput);

	maxNumberOfTrainingIterations = 10000;


}


void GEKModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;


}

void GEKModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void GEKModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());
	hyperparameters_filename = filename;

}



void GEKModel::initializeSurrogateModel(void){

	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();



	printf("Initializing settings for the GEK model...\n");

	modelID = GRADIENT_ENHANCED_KRIGING;
	ifHasGradientData = true;

	readData();
	normalizeData();

	numberOfHyperParameters = dim;

	GEK_weights =zeros<vec>(numberOfHyperParameters);


	/* regularization term */
	epsilonGEK = 0.0;

	/* check if two sample are too close to each other */
	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec sample1 = data.getRowX(i);


		for(unsigned int j=i+1; j<numberOfSamples; j++){

			rowvec sample2 = data.getRowX(j);

			if(checkifTooCLose(sample1, sample2)) {

				printf("ERROR: Two samples in the training data are too close to each other!\n");
				abort();
			}
		}
	}

	sigmaSquared = 0.0;
	beta0 = 0.0;


	correlationMatrixDot = zeros(numberOfSamples*(dim+1),numberOfSamples*(dim+1));
	upperDiagonalMatrixDot= zeros<mat>(numberOfSamples*(dim+1),numberOfSamples*(dim+1));

	R_inv_ys_min_beta = zeros<vec>(numberOfSamples*(dim+1));
	R_inv_F= zeros<vec>(numberOfSamples*(dim+1));
	vectorOfF= zeros<vec>(numberOfSamples*(dim+1));

	for(unsigned int i=0; i<numberOfSamples; i++) {

		vectorOfF(i)=1.0;
	}


	yGEK = zeros<vec>(numberOfSamples*(dim+1));

	/* first N entries are the functional values */

	vec y = data.getOutputVector();

	for(unsigned int i=0; i<numberOfSamples; i++){

		yGEK(i) =y(i);

	}

	/* rest entries are the gradients (scaled to the feature space) */

	mat gradientData = data.getGradientMatrix();

	Bounds boxConstraints = data.getBoxConstraints();

	for(unsigned int i=0; i<dim; i++){

		vec gradx = gradientData.col(i);

		for(unsigned int j=0; j<numberOfSamples; j++){

			double xmin = boxConstraints.getLowerBound(i);
			double xmax = boxConstraints.getUpperBound(i);

			yGEK(numberOfSamples+i*numberOfSamples+j) = gradx(j)*( xmax - xmin )*dim;



		}
	}

#if 0
	printVector(y,"y");
	printMatrix(gradientData,"gradientData");
	printVector(yGEK,"yGEK");
#endif

	ifInitialized = true;

	std::cout << "GEK model initialization is done...\n";


}

void GEKModel::printSurrogateModel(void) const{

	data.print();

	printVector(GEK_weights,"GEK_weights");

}

void GEKModel::printHyperParameters(void) const{

	printVector(GEK_weights,"GEK_weights");


}

void GEKModel::saveHyperParameters(void) const{



}
void GEKModel::loadHyperParameters(void){


}
void GEKModel::train(void){

	if(!ifInitialized){

		initializeSurrogateModel();

	}

	KrigingModel auxModelForTraining(name);
	auxModelForTraining.setGradientsOn();

	auxModelForTraining.initializeSurrogateModel();
	auxModelForTraining.setNumberOfTrainingIterations(maxNumberOfTrainingIterations);
	auxModelForTraining.printSurrogateModel();

	auxModelForTraining.train();

	GEK_weights = auxModelForTraining.getTheta();
#if 1
	printVector(GEK_weights,"GEK_weights");
#endif

	updateAuxilliaryFields();

}


double GEKModel::interpolate(rowvec x ) const {


	vec r = computeCorrelationVectorDot(x);
#if 0
	printVector(r,"r");
	cout<<"beta0 ="<<beta0<<"\n";
#endif


	double fGEK = beta0 + dot(r,R_inv_ys_min_beta);

	return fGEK;

}
void GEKModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const {

	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();

	*ftildeOutput =  interpolate(xp);

	vec R_inv_r(numberOfSamples*(dim+1));

	vec r = computeCorrelationVectorDot(xp);

	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_r, r);


	*sSqrOutput = sigmaSquared*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_F) -1.0 ),2.0)) / (dot(vectorOfF,R_inv_F) ) );

}


/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^i_k (for GEK)
 *
 *
 * */

double GEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double result;

	double R = computeCorrelation(x_i, x_j, theta);
	result= -2.0*theta(k)* (x_i(k)-x_j(k))* R;
	return result;
}





/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^j_k (for GEK)
 *
 *
 *
 * */

double GEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta = GEK_weights;
	double result = 0.0;
	double R = computeCorrelation(x_i, x_j, theta);

	result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;

	return result;
}


/*
 *
 * second derivative of R(x^i,x^j) w.r.t. x^i_l and x^j_k (hand derivation)
 * (for GEK)
 *
 * */

double GEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

	double dx;
	vec theta = GEK_weights;
	double R = computeCorrelation(x_i, x_j, theta);

	if (k == l){

		dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*R;
	}
	if (k != l) {

		dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*R;
	}

	return dx;
}

double GEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	for (unsigned int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2.0);

	}

	return exp(-sum);
}


/* implementation according to the Forrester book */
void GEKModel::computeCorrelationMatrixDot(void) {

	unsigned int numberOfSamples = data.getNumberOfSamples();
	mat X = data.getInputMatrix();

	vec theta = GEK_weights;
	int k = X.n_cols;

	mat Psi=zeros(numberOfSamples,numberOfSamples);
	mat PsiDot=zeros(numberOfSamples,numberOfSamples);


	mat Rfull;

	for(int row = -1; row < k; row++){

		if(row == -1){ /* first row */

			for(unsigned int i=0; i<numberOfSamples;i++){
				for(unsigned int j=i+1;j<numberOfSamples;j++){

					Psi(i,j)= computeCorrelation(X.row(i), X.row(j), theta);

				}
			}

			Psi = Psi+ trans(Psi)+ eye(numberOfSamples,numberOfSamples);

			Rfull=Psi;


			PsiDot=zeros(numberOfSamples,numberOfSamples);
			for(int l=0;l<k; l++){


				for(unsigned int i=0; i<numberOfSamples;i++){
					for(unsigned int j=0;j<numberOfSamples;j++){
						PsiDot(i,j)=2.0*theta(l)* (X(i,l)-X(j,l))*Psi(i,j);

					}
				}
				Rfull = join_rows(Rfull,PsiDot);

			}

		}

		else{ /* other rows */

			mat Rrow;

			PsiDot=zeros(numberOfSamples,numberOfSamples);

			for(unsigned int i=0; i<numberOfSamples;i++){
				for(unsigned int j=0;j<numberOfSamples;j++){

					PsiDot(i,j)=-2.0*theta(row)* (X(i,row)-X(j,row))*Psi(i,j);

				}
			}

			Rrow = PsiDot;

			for(int l=0; l<k;l++){
				mat PsiDot2=zeros(numberOfSamples,numberOfSamples);

				if(l == row){
					for(unsigned int i=0; i<numberOfSamples;i++){
						for(unsigned int j=0;j<numberOfSamples;j++){
							PsiDot2(i,j)=
									(2.0*theta(l)-4.0*theta(l)*theta(l)* pow((X(i,l)-X(j,l)),2.0))*Psi(i,j);

						}
					}

				}

				else{


					for(unsigned int i=0; i<numberOfSamples;i++){
						for(unsigned int j=0;j<numberOfSamples;j++){

							PsiDot2(i,j)=
									(-4.0*theta(row)*theta(l)*(X(i,row)-X(j,row))*(X(i,l)-X(j,l)))*Psi(i,j);

						}
					}
				}

				Rrow = join_rows(Rrow,PsiDot2);
			}

			Rfull = join_cols(Rfull,Rrow);
		}

	} /* end of for loop for rows */



	correlationMatrixDot  = Rfull + epsilonGEK * eye(numberOfSamples*(k+1),numberOfSamples*(k+1));



} /* end of compute_R_matrix_GEK */

vec GEKModel::computeCorrelationVectorDot(rowvec x) const{


	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();
	mat X = data.getInputMatrix();

	vec r(numberOfSamples*(dim+1));

	vec theta = GEK_weights;


	int count = 0;
	for(unsigned int i=0;i<numberOfSamples;i++){

		r(count) = computeCorrelation(x, X.row(i), theta);
		count++;
	}

	for(unsigned int i=0;i<dim;i++){
		for(unsigned int j=0;j<numberOfSamples;j++){

			r(count) = computedR_dxj(x, X.row(j),i);
			count++;

		}
	}

	return r;

}


void GEKModel::updateAuxilliaryFields(void){

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

#if 0
	cout<<"Updating auxiliary variables of the GEK model\n";
#endif
	vec ys = yGEK;

	computeCorrelationMatrixDot();



#if 0

	cout<<"\nCorrelation matrix\n";
	for(unsigned int i=0; i< correlationMatrixDot.n_cols; i++ ){

		vec Rdotcol = correlationMatrixDot.col(i);
		cout<<"Column: "<<i<<"\n";
		printVector(Rdotcol);


	}

#endif





	/* Cholesky decomposition R = LDL^T */


	correlationMatrixDot.save("correlationMatrix.csv",csv_ascii);


	int cholesky_return = chol(upperDiagonalMatrixDot, correlationMatrixDot);

	if (cholesky_return == 0) {

		printf("ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}





#if 0
	printMatrix(upperDiagonalMatrixDot,"upperDiagonalMatrixDot");
#endif

	vec R_inv_ys(N*(dim+1)); R_inv_ys.fill(0.0);


	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys, ys);    /* solve R x = ys */





	R_inv_F = zeros(N*(dim+1));


	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_F, vectorOfF);      /* solve R x = F */


	beta0 = (1.0/dot(vectorOfF,R_inv_F)) * (dot(vectorOfF,R_inv_ys));

	vec ys_min_betaF = ys - beta0*vectorOfF;



	/* solve R x = ys-beta0*I */



	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys_min_beta , ys_min_betaF);




	sigmaSquared = (1.0 / (N*(dim+1))) * dot(ys_min_betaF, R_inv_ys_min_beta);


}


void GEKModel::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{

	std::cout<<"ERROR: GEKModel::calculateExpectedImprovement is not implemented yet!\n";
	abort();


}


void GEKModel::addNewSampleToData(rowvec newsample){


	std::cout<<"ERROR: GEKModel::addNewSampleToData is not implemented yet!\n";
	abort();


}
