#include "kernel_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;

double calcMetric(rowvec &xi,rowvec &xj, mat M){

	rowvec diff= xi-xj;

	colvec diffT= trans(diff);

	return dot(diffT,M*diff);

}


double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M){

/* calculate distance between xi and xj with the matrix M */
	double metricVal = calcMetric(xi,xj,M);

	return (1.0/(sigma*datum::pi))*exp(-metricVal/(2*sigma*sigma));


}

double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma){

	int d = y.size();

	vec kernelVal(d);
	vec weight(d);
	double kernelSum=0.0;
	double yhat=0.0;

	for(int i=0; i<d; i++){

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi,xp,sigma,M);
		kernelSum+=kernelVal(i);
	}

	for(int i=0; i<d; i++){

		weight(i)=kernelVal(i)/kernelSum;
		yhat = y(i)*weight(i);
	}

	return yhat;


}

