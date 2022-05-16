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

#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "sgek.hpp"
#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

//int total_number_of_function_evals_GEK;
//double population_overall_max_GEK = -10E14;
//int population_overall_max_tread_id_GEK = -1;

SGEKModel::SGEKModel():SurrogateModel(){}


SGEKModel::SGEKModel(std::string nameInput):SurrogateModel(nameInput){

	modelID = SLICED_GRADIENT_ENHANCED_KRIGING;
	setName(nameInput);
	setNameOfHyperParametersFile(nameInput);

	maxNumberOfTrainingIterations = 10000;


}


void SGEKModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;


}

void SGEKModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void SGEKModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());
	hyperparameters_filename = filename;

}



void SGEKModel::initializeSurrogateModel(void){

	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();

	printf("Initializing settings for the SGEK model...\n");

	modelID = SLICED_GRADIENT_ENHANCED_KRIGING;
	ifHasGradientData = true;

	readData();                 // Modified by Kai
	normalizeData();            // Modified by Kai

	numberOfHyperParameters = dim;

	GEK_weights =zeros<vec>(numberOfHyperParameters);


	/* regularization term */
	epsilonGEK = 0.0;

	/* check if two samples are too close to each other */
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
			//yGEK(numberOfSamples+i*numberOfSamples+j) = gradx(j)*( xmax - xmin );


		}
	}

#if 0
	printVector(y,"y");
	printMatrix(gradientData,"gradientData");
	printVector(yGEK,"yGEK");
#endif

	ifInitialized = true;

	std::cout << "GEK model initialization is done...\n";

	snum = 10;
    slicing(snum);

}

void SGEKModel::printSurrogateModel(void) const{

	data.print();

	printVector(GEK_weights,"GEK_weights");

}

void SGEKModel::printHyperParameters(void) const{

	printVector(GEK_weights,"GEK_weights");


}

void SGEKModel::saveHyperParameters(void) const{



}
void SGEKModel::loadHyperParameters(void){


}
void SGEKModel::train(void){

	if(!ifInitialized){

		initializeSurrogateModel();

	}

	unsigned int dim = data.getDimension();

	vec hyper_l = {0.0005, 0.2, 0.0005};   // lower bound
	vec hyper_u = {2.5, 1, 2.5};           // upper bound

	num = 10;

	cout <<  "Start hyper-parameter optimization" << endl;

	boxmin(hyper_l,hyper_u,num);   // Hooke Jeeves algorithm for hyper-parameter optimization

	GEK_weights = getTheta();
	likelihood  = getLikelihood();

	// original_likelihood_function(GEK_weights);

	cout <<  "The optimal hyper-parameter is " << GEK_weights << endl;
	cout <<  "The optimal likelihood is " << likelihood << endl;

#if 1
	printVector(GEK_weights,"GEK_weights");
#endif

	updateAuxilliaryFields();

}


void SGEKModel::slicing(unsigned int snum ){

	dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();
	unsigned int mn = N*(dim+1);

	mat X = data.getInputMatrix();

	gradient = reshape(yGEK.subvec(N,mn-1),N,dim);

	sensitivity  = mean(gradient.t() % gradient.t(),1);       // resort data based on the global sensitivity analysis results

	sensitivity  = sensitivity / sum(sensitivity);

	uvec ind = sort_index(sensitivity,"descend");

	X = X.cols(ind);

	gradient = gradient.cols(ind);

	uvec inds = sort_index(X.col(ind(1)),"descend");   // sort sample and ready for slicing

	unsigned int Hn = floor(N/snum); unsigned int  re = N % snum;     // round down

    field<uvec> index1(snum);              // store the index of sampling sites within each slice

	if (re == 0){
	    for (unsigned int k=0; k<snum; k++){
	        index1(k) = inds.subvec(k*Hn,(k+1)*Hn-1);

        }
    }
	else {
	   for (unsigned int k=0; k<snum; k++){
	      if (k < re)
	         index1(k) = inds.subvec(k*(Hn+1),(k+1)*(Hn+1)-1);
	      else
	         index1(k) = inds.subvec(re*(Hn+1)+(k-re)*Hn,re*(Hn+1)+(k-re+1)*Hn-1);
	   }
	}

	index = index1;


}

void SGEKModel::original_likelihood_function(vec alpha){ //  Sliced likelihood function

	vec theta = alpha(1)*pow(sensitivity,alpha(2))+alpha(3);

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();
	unsigned int mn = N*(dim+1);
	mat X = data.getInputMatrix();

	correlationMatrixDot = correlationfunction.corrbiquadspline_gekriging(X,theta);

	upperDiagonalMatrixDot = chol(correlationMatrixDot);

	long double logdetR = 2*sum(log(diagvec(upperDiagonalMatrixDot)));

	vec R_inv_ys(mn); R_inv_ys.fill(0.0);

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys, yGEK);    /* solve R x = ys */

	R_inv_F = zeros(mn);

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_F, vectorOfF);      /* solve R x = F */

	beta0 = (1.0/dot(vectorOfF,R_inv_F)) * (dot(vectorOfF,R_inv_ys));

	vec ys_min_betaF = yGEK - beta0*vectorOfF;

	/* solve R x = ys-beta0*I */

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys_min_beta , ys_min_betaF);

	sigmaSquared = (1.0 / (mn)) * dot(ys_min_betaF, R_inv_ys_min_beta);

	 likelihood = mn * log(sigmaSquared) + logdetR;

}


double SGEKModel::sliced_likelihood_function(vec alpha){ //  Sliced likelihood function

	vec theta = alpha(0)*pow(sensitivity,alpha(1))+alpha(2);

    dim = data.getDimension();
	unsigned int N   = data.getNumberOfSamples();
	unsigned int mn  = N*(dim+1);

	mat X = data.getInputMatrix();

	mat yt = reshape(yGEK,N,dim+1);  // response and gradient

	field<mat>  X_2(snum-1);
	field<mat>  upperDiagonalMatrixDot_2(snum-1);
	field<mat>  correlationMatrixDot_2(snum-1);

	field<vec>  yGEK_2(snum-1);
	field<vec>  R_inv_F_2(snum-1);
	field<vec>  vectorOfF_2(snum-1);
	field<vec>  R_inv_ys_2(snum-1);
	field<vec>  R_inv_ys_min_beta_2(snum-1);
	field<vec>  ys_min_betaF_2(snum-1);
	field<uvec> index_2(snum-1);

	vec logdetR_2(snum-1);
	vec n_2(snum-1);
	vec nom_2(snum-1);
	vec denom_2(snum-1);
	vec sig_2(snum-1);


	field<mat>  X_1(snum-2);
	field<mat>  upperDiagonalMatrixDot_1(snum-2);
	field<mat>  correlationMatrixDot_1(snum-2);

	field<vec>  yGEK_1(snum-2);
	field<vec>  R_inv_F_1(snum-2);
	field<vec>  vectorOfF_1(snum-2);
	field<vec>  R_inv_ys_1(snum-2);
	field<vec>  R_inv_ys_min_beta_1(snum-2);
	field<vec>  ys_min_betaF_1(snum-2);
	field<uvec> index_1(snum-2);

	vec logdetR_1(snum-2);
	vec n_1(snum-2);
	vec nom_1(snum-2);
	vec denom_1(snum-2);
	vec sig_1(snum-2);

	for (unsigned int k=0; k< snum-1; k++){

		n_2(k) = size(index(k),0)+size(index(k+1),0);

		index_2(k) = join_cols(index(k),index(k+1));

		X_2(k) = X.rows(index_2(k));

		upperDiagonalMatrixDot_2(k) = chol(correlationfunction.corrbiquadspline_gekriging(X_2(k),theta));

		logdetR_2(k) = 2*sum(log(diagvec(upperDiagonalMatrixDot_2(k))));

		R_inv_ys_2(k) = zeros(n_2(k)*(dim+1));

		yGEK_2(k) =  reshape(yt.rows(index_2(k)),n_2(k)*(dim+1),1);

		solveLinearSystemCholesky(upperDiagonalMatrixDot_2(k), R_inv_ys_2(k), yGEK_2(k));

		R_inv_F_2(k) = zeros(n_2(k)*(dim+1));

		vectorOfF_2(k) = join_cols(ones(n_2(k)),zeros(n_2(k)*dim));

		solveLinearSystemCholesky(upperDiagonalMatrixDot_2(k), R_inv_F_2(k),vectorOfF_2(k));

		nom_2(k)  =  dot(vectorOfF_2(k),R_inv_ys_2(k));

		denom_2(k)= dot(vectorOfF_2(k),R_inv_F_2(k));

	}

	for (unsigned int k=0; k< snum-2; k++){

			n_1(k) = size(index(k+1),0);

			X_1(k) = X.rows(index(k+1));

			upperDiagonalMatrixDot_1(k) = chol(correlationfunction.corrbiquadspline_gekriging(X_1(k),theta));

			logdetR_1(k) = 2*sum(log(diagvec(upperDiagonalMatrixDot_1(k))));

			R_inv_ys_1(k) = zeros(n_1(k)*(dim+1));

			yGEK_1(k) =  reshape(yt.rows(index(k+1)),n_1(k)*(dim+1),1);

			solveLinearSystemCholesky(upperDiagonalMatrixDot_1(k), R_inv_ys_1(k), yGEK_1(k));

			R_inv_F_1(k) = zeros(n_1(k)*(dim+1));

			vectorOfF_1(k) = join_cols(ones(n_1(k)),zeros(n_1(k)*dim));

			solveLinearSystemCholesky(upperDiagonalMatrixDot_1(k), R_inv_F_1(k),vectorOfF_1(k));

			nom_1(k)  =  dot(vectorOfF_1(k),R_inv_ys_1(k));

			denom_1(k) = dot(vectorOfF_1(k),R_inv_F_1(k));

		}

	 beta0 = (sum(nom_2)-sum(nom_1))/(sum(denom_2)-sum(denom_1));


	 for (unsigned int k=0; k< snum-2; k++){

		 ys_min_betaF_2(k) = yGEK_2(k)- beta0*vectorOfF_2(k);

		 R_inv_ys_min_beta_2(k) = zeros(n_2(k)*(dim+1));

		 solveLinearSystemCholesky(upperDiagonalMatrixDot_2(k), R_inv_ys_min_beta_2(k), ys_min_betaF_2(k));

         sig_2(k) = dot(ys_min_betaF_2(k), R_inv_ys_min_beta_2(k));

         ys_min_betaF_1(k) = yGEK_1(k)- beta0*vectorOfF_1(k);

         R_inv_ys_min_beta_1(k) = zeros(n_1(k)*(dim+1));

         solveLinearSystemCholesky(upperDiagonalMatrixDot_1(k), R_inv_ys_min_beta_1(k), ys_min_betaF_1(k));

         sig_1(k) = dot(ys_min_betaF_1(k), R_inv_ys_min_beta_1(k));
	 }

	  ys_min_betaF_2(snum-2) = yGEK_2(snum-2)- beta0*vectorOfF_2(snum-2);

	  R_inv_ys_min_beta_2(snum-2) = zeros(n_2(snum-2)*(dim+1));

	  solveLinearSystemCholesky(upperDiagonalMatrixDot_2(snum-2), R_inv_ys_min_beta_2(snum-2), ys_min_betaF_2(snum-2));

	  sig_2(snum-2) = dot(ys_min_betaF_2(snum-2), R_inv_ys_min_beta_2(snum-2));

	  sigmaSquared = (sum(sig_2)-sum(sig_1))/mn;

      likelihood = mn * log(sigmaSquared) + sum(logdetR_2) - sum(logdetR_1);

      return likelihood;

}


 double SGEKModel::interpolate(rowvec x ) const {


	vec r = computeCorrelationVectorDot(x);
#if 0
	printVector(r,"r");
	cout<<"beta0 ="<<beta0<<"\n";
#endif


	double fGEK = beta0 + dot(r,R_inv_ys_min_beta);

	return fGEK;

}

/* mat SGEKModel::interpolate_all(mat xtest) {

	mat X = data.getInputMatrix();
	vec theta = GEK_weights;

	mat r = correlationfunction.corrbiquadspline_gekriging_vec(xtest,X,theta);

   // mat r = computeCorrelationVectorDot(x);
   //vec fGEK = beta0 + dot(r,R_inv_ys_min_beta);

	vec fGEK;

	for (unsigned int k = 0; k < dim; k++) {

		fGEK(k) = beta0 + dot(r.row(k),R_inv_ys_min_beta);

	}

	return fGEK;
}*/

void SGEKModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const {

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
 * derivative of R(x^i,x^j) w.r.t. x^i_k (for GEK)
 *
 *
 * */

/* double SGEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double result;

	double R = computeCorrelation(x_i, x_j, theta);
	result= -2.0*theta(k)* (x_i(k)-x_j(k))* R;
	return result;
}*/

double SGEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double result;
    double xi;
    double ui;
	double R = computeCorrelation(x_i, x_j, theta);

	/*result= -2.0*theta(k)* (x_i(k)-x_j(k))* R; */

	xi = fabs(x_i(k)-x_j(k))*theta(k);       /* Modified by Kai Cheng */
	ui  = sign(x_i(k)-x_j(k))*theta(k);
	if (xi <= 0.4)
		 { result = -ui*(-30*xi + 105*pow(xi,2) - 195.0/2*pow(xi,3))/(1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4))*R;}
    else if (xi < 1)
		 { result = -ui*(-20.0/3 + 20*xi- 20*pow(xi,2) + 20.0/3*pow(xi,3))/(5/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}
	else
		 { result = 0;}
	return result;
}

/*
 * derivative of R(x^i,x^j) w.r.t. x^j_k (for GEK)
 * */

/* double SGEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta = GEK_weights;
	double result = 0.0;
	double R = computeCorrelation(x_i, x_j, theta);

	result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;

	return result;
}*/

double SGEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta = GEK_weights;
	double result = 0.0;
	double R = computeCorrelation(x_i, x_j, theta);
    double xi;
    double ui;

    /* result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;*/

	xi = fabs(x_i(k)-x_j(k))*theta(k);     /* Modified by Kai Cheng */
	ui  = sign(x_i(k)-x_j(k))*theta(k);
	if (xi <= 0.4)
       { result = -ui*(-30*xi + 105*pow(xi,2) - 195.0/2*pow(xi,3))/(1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4))*R;}
    else if (xi < 1)
	   { result = -ui*(-20.0/3 + 20*xi- 20*pow(xi,2) + 20.0/3*pow(xi,3))/(5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}
	else
	   { result = 0;}

	return result;
}


/*
 *
 * second derivative of R(x^i,x^j) w.r.t. x^i_l and x^j_k (hand derivation)
 * (for GEK)
 *
 * */

/* double SGEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

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
}*/

double SGEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

	double dx;
	double xi;
	double xi1;
	double ui1;
	double xi2;
	double ui2;
	vec theta = GEK_weights;
	double R = computeCorrelation(x_i, x_j, theta);

	if (k == l){

   /* dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*R;*/

	 xi = fabs(x_i(k)-x_j(k))*theta(k);
	 if (xi <= 0.4)
		 { dx = -(-30 + 210*xi - 585.0/2*pow(xi,2))*pow(theta(k),2)/(1-15*pow(xi,2) + 35*pow(xi,3)-195.0/8*pow(xi,4))*R;}
	 else if (xi < 1)
		 { dx= -(20 - 40*xi + 20*pow(xi,2))*pow(theta(k),2)/(5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}
     else
		 { dx = 0;}

	}
	if (k != l) {

		/* dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*R;*/
		xi1  = fabs(x_i(k)-x_j(k))*theta(k);     /* Modified by Kai Cheng */
		ui1  = sign(x_i(k)-x_j(k))*theta(k);
		xi2  = fabs(x_i(l)-x_j(l))*theta(l);
		ui2  = sign(x_i(l)-x_j(l))*theta(l);

		if (xi1 <= 0.4 && xi2 <= 0.4)
			 { dx = -(ui1*(-30*xi1 + 105*pow(xi1,2) - 195.0/2*pow(xi1,3))*ui2*(-30*xi2 + 105*pow(xi2,2) - 195.0/2*pow(xi2,3)))/((1 - 15*pow(xi1,2) + 35*pow(xi1,3) - 195.0/8*pow(xi1,4))*(1 - 15*pow(xi2,2) + 35*pow(xi2,3) - 195.0/8*pow(xi2,4)))*R;}
		else if (xi1 < 1 && xi2 <= 0.4)
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*pow(xi1,2) + 20.0/3*pow(xi1,3))*ui2*(-30*xi2 + 105*pow(xi2,2) - 195.0/2*pow(xi2,3)))/((5.0/3 - 20.0/3*xi1 + 10*pow(xi1,2) - 20.0/3*pow(xi1,3) + 5.0/3*pow(xi1,4))*(1 - 15*pow(xi2,2) + 35*pow(xi2,3) - 195.0/8*pow(xi2,4)))*R;}
		else if(xi1 < 1 && xi2 < 1 )
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*pow(xi1,2) + 20.0/3*pow(xi1,3))*ui2*(-20.0/3 + 20*xi2- 20*pow(xi2,2) + 20.0/3*pow(xi2,3)))/((5/3 - 20.0/3*xi1 + 10*pow(xi1,2) - 20.0/3*pow(xi1,3) + 5.0/3*pow(xi1,4))*(5.0/3 - 20.0/3*xi2 + 10*pow(xi2,2) - 20.0/3*pow(xi2,3) + 5.0/3*pow(xi2,4)))*R;}
		else if (xi1 < 0.4 && xi2 < 1 )
		     { dx = -(ui1*(-30*xi1 + 105*pow(xi1,2) - 195.0/2*pow(xi1,3))*ui2*(-20.0/3 + 20*xi2- 20*pow(xi2,2) + 20.0/3*pow(xi2,3)))/((1 - 15*pow(xi1,2) + 35*pow(xi1,3) - 195.0/8*pow(xi1,4))*(5.0/3 - 20.0/3*xi2 + 10*pow(xi2,2) - 20.0/3*pow(xi2,3) + 5.0/3*pow(xi2,4)))*R;}
		else
			 { dx = 0;}
	}

	return dx;
}

/* double SGEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	for (unsigned int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2.0);

	}

	return exp(-sum);
}*/

double SGEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	double xi = 0.0;
	vec ss(dim);

	for (unsigned int k = 0; k < dim; k++) {

		  xi = fabs(x_i(k) - x_j(k))*theta(k);

		  if (xi <= 0.4)
			  {ss(k) = 1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4);}
		  else if (xi < 1)
			  {ss(k) = 5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4);}
		  else
			  {ss(k) = 0;}
        }

	return prod(ss);
}

/* implementation according to the Forrester book */
/* void SGEKModel::computeCorrelationMatrixDot(vec theta) {

	unsigned int numberOfSamples = data.getNumberOfSamples();
	mat X = data.getInputMatrix();

	//vec theta = GEK_weights;
	int k = X.n_cols;

	mat Psi=zeros(numberOfSamples,numberOfSamples);
	mat PsiDot=zeros(numberOfSamples,numberOfSamples);


	mat Rfull;

	for(int row = -1; row < k; row++){

		if(row == -1){ /* first row

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

		else{ /* other rows

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

	} /* end of for loop for rows



	correlationMatrixDot  = Rfull + epsilonGEK * eye(numberOfSamples*(k+1),numberOfSamples*(k+1));



} */ /* end of compute_R_matrix_GEK */


vec SGEKModel::computeCorrelationVectorDot(rowvec x) const{


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


void SGEKModel::updateAuxilliaryFields(void){

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

#if 0
	cout<<"Updating auxiliary variables of the GEK model\n";
#endif
	vec ys = yGEK;
	vec theta = GEK_weights;
	mat X = data.getInputMatrix();
	// computeCorrelationMatrixDot(theta);

	correlationMatrixDot = correlationfunction.corrbiquadspline_gekriging(X,theta);


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


void SGEKModel::addNewSampleToData(rowvec newsample){


	/* avoid points that are too close to each other */

		mat rawData = data.getRawData();

		bool flagTooClose= checkifTooCLose(newsample, rawData);


		if(!flagTooClose){

			appendRowVectorToCSVData(newsample, filenameDataInput);

			updateModelWithNewData();

		}
		else{

			std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

		}

}

void SGEKModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();

	unsigned int N = data.getNumberOfSamples();
	unsigned int dim = data.getDimension();

	normalizeData();

	correlationMatrixDot.set_size(N*(dim+1),N*(dim+1));
	correlationMatrixDot.fill(0.0);
	upperDiagonalMatrixDot.set_size(N*(dim+1),N*(dim+1));
	upperDiagonalMatrixDot.fill(0.0);
	R_inv_ys_min_beta.set_size(N*(dim+1));
	R_inv_ys_min_beta.fill(0.0);
	R_inv_F.set_size(N*(dim+1));
	R_inv_F.fill(0.0);
	vectorOfF.set_size(N*(dim+1));
	vectorOfF.fill(1.0);


	updateAuxilliaryFields();

}

void SGEKModel::resetDataObjects(void){

	correlationMatrixDot.reset();
	upperDiagonalMatrixDot.reset();
	R_inv_F.reset();
	R_inv_ys_min_beta.reset();
	vectorOfF.reset();

	beta0 = 0.0;
	sigmaSquared = 0.0;

}

void SGEKModel::calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const{

	double ftilde = 0.0;
		double ssqr   = 0.0;

		interpolateWithVariance(currentDesign.dv,&ftilde,&ssqr);

		#if 0
			printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
		#endif

		double	sigma = sqrt(ssqr)	;

		#if 0
			printf("standart_ERROR = %15.10f\n",sigma);
		#endif

		double expectedImprovementValue = 0.0;

		if(fabs(sigma) > EPSILON){

			double ymin = data.getMinimumOutputVector();
			double	Z = (ymin - ftilde)/sigma;
		#if 0
			printf("Z = %15.10f\n",Z);
			printf("ymin = %15.10f\n",ymin);
		#endif


			expectedImprovementValue = (ymin - ftilde)*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
		}
		else{

			expectedImprovementValue = 0.0;

			}
		#if 0
			printf("expectedImprovementValue = %20.20f\n",expectedImprovementValue);
		#endif

			currentDesign.objectiveFunctionValue = ftilde;
			currentDesign.valueExpectedImprovement = expectedImprovementValue;


}


void SGEKModel::boxmin(vec hyper_l, vec hyper_u, int num){

    dim_a = 3;

	mat hyper = ones(dim_a,num);
	vec likeli_value = ones(num);

	vec log_ub = log10(hyper_u);
	vec log_lb = log10(hyper_l);
    vec random;  random.randu(num);

	for (unsigned int i=0;i<dim_a;i++){
	  for (unsigned int j=0;j<num;j++){
		  hyper(i,j) = pow(10,random(j)*(log_ub(i)-log_lb(i))+log_lb(i));
	  }
    }

	hyper_lb  = hyper_l;                  // lower bound
	hyper_up  = hyper_u;                  // upper bound


	 //#pragma omp parallel for
	for (unsigned int kk=0;kk<num;kk++){

	  start(hyper.col(kk),hyper_lb,hyper_up);

	  int kmax;

	  if (dim_a < 2)
	      { kmax = 2;}
      else
	      { kmax = std::min(dim_a,4);}

	  for (unsigned int k = 0; k < kmax; k++){

	   vec hyper1 = hyper_cur;

	   explore(hyper_cur,likelihood_cur);

	   move(hyper1,hyper_cur,likelihood_cur);
	 }

	  hyper.col(kk) = getAlpha();
	  likeli_value(kk) = getLikelihood();

	}

	uword i = likeli_value.index_min();

	likelihood_cur = likeli_value(i);
	hyper_cur = hyper.col(i);

}

void SGEKModel::start(vec hyper_in, vec hyper_l, vec hyper_u){

	  vec m = linspace(1,dim_a,dim_a)/(dim_a+2);
	  increment = zeros(dim_a);

	  for (unsigned int k = 0; k < dim_a; k++){
		  increment(k) = pow(2,m(k));
	   }

	  ind_increment = find(increment != 1);

	  hyper_cur = hyper_in;

	  likelihood_cur = sliced_likelihood_function(hyper_cur);

	  numberOfIteration = 0;
      hyperoptimizationHistory = zeros(dim_a+2,200*dim);

	  hyperoptimizationHistory.col(numberOfIteration) = join_cols(hyper_cur, vec { likelihood_cur, 1.0} );

}

void SGEKModel::explore(vec hyper_1, double likelihood_1){

	unsigned int j; double DD;  unsigned int atbd;

    hyper_cur = hyper_1; likelihood_cur = likelihood_1;

	for (unsigned int k = 0; k < size(ind_increment,0); k++){

	   j = ind_increment(k);
	   hyper_par = hyper_cur;
       DD = increment(j);

       if (hyper_cur(j) == hyper_up(j)){

    	   atbd = 1;
    	   hyper_par(j) =  hyper_cur(j)/sqrt(DD); }

       else if (hyper_cur(j) == hyper_lb(j)){

    	   atbd = 1;
    	   hyper_par(j) =  hyper_cur(j)*sqrt(DD); }

       else  {

    	   atbd = 0;
    	   hyper_par(j) = std::min(hyper_up(j),hyper_cur(j)*DD);
       }

      // cout << hyper_par << endl;

       likelihood = sliced_likelihood_function(hyper_par);

      // cout << likelihood << endl;

       numberOfIteration++;
       hyperoptimizationHistory.col(numberOfIteration)= join_cols(hyper_par, vec { likelihood, 2});

       if (likelihood < likelihood_cur){
            likelihood_cur = likelihood;
            hyper_cur = hyper_par;  }
       else  {

    	    hyperoptimizationHistory(dim_a+1,numberOfIteration)= -2;

    	   if (!atbd) {
    		    hyper_par(j) = std::max(hyper_lb(j),hyper_cur(j)/DD);
    	        likelihood = sliced_likelihood_function(hyper_par);

    	        numberOfIteration++;
    	        hyperoptimizationHistory.col(numberOfIteration)=join_cols(hyper_par, vec { likelihood, 2});

    	        if (likelihood < likelihood_cur){
    	        	likelihood_cur = likelihood;
    	        	hyper_cur = hyper_par;  }
    	        else
    	            hyperoptimizationHistory(dim_a+1,numberOfIteration)= -2;
    	    }
	     }
	  }
}

void SGEKModel::move(vec hyper_old,vec hyper_new, double likelihood_new){

	   vec v  = hyper_new/hyper_old;
	   vec v1 = v-ones(dim_a,1);

      // cout << "move step" << endl;

       if (v1.is_zero()){

    	   vec ind = linspace(1,dim_a,dim_a);
    	   ind(dim_a-1) = 0;

    	   for (unsigned int k = 0; k < dim_a; k++){
    	  	    increment(k) = pow(increment(ind(k)),0.2);
    	    }

    	   likelihood_cur = likelihood_new;
    	   hyper_cur = hyper_new;

            return ;
        }

        unsigned int rept = 1;   likelihood_cur = likelihood_new;  hyper_cur = hyper_new;

        while (rept){

		   hyper_par = min(join_rows(hyper_up,max(join_rows(hyper_lb,hyper_new % v),1)),1);
		   likelihood = sliced_likelihood_function(hyper_par);
		   numberOfIteration++;
		   hyperoptimizationHistory.col(numberOfIteration)= join_cols(hyper_par, vec { likelihood, 3});

		   if (likelihood < likelihood_cur){

			   hyper_cur = hyper_par;
			   likelihood_cur = likelihood;
			   v = v % v;
		   }

		   else {
			   hyperoptimizationHistory(dim_a+1,numberOfIteration)= -3;
			   rept = 0;

		   }

		   if (size(find(hyper_par - hyper_up),0)+size(find(hyper_par - hyper_up),0) < 2*dim_a)
			    rept  =  0;
        }

         vec ind = linspace(1,dim_a,dim_a);
         ind(dim_a-1) = 0;

         for (unsigned int k = 0; k < dim_a; k++){
          	      increment(k) = pow(increment(ind(k)),0.25);
          }
}

vec SGEKModel::getTheta(void) const{

	return hyper_cur(0)*pow(sensitivity,hyper_cur(1))+hyper_cur(2);

}

vec SGEKModel::getAlpha(void) const{

	return hyper_cur;

}

double SGEKModel::getLikelihood(void) const{

	return likelihood_cur;


}
