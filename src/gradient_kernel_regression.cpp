#include <armadillo>
#include <iostream>

#include "kernel_regression.hpp"
#include "gradient_kernel_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"

using namespace std;
using namespace arma;




GradientKernelRegressionModel::GradientKernelRegressionModel():KernelRegressionModel(){}


GradientKernelRegressionModel::GradientKernelRegressionModel(std::string name):KernelRegressionModel(name){

	modelID = GRADIENT_ENHANCED_KERNEL_REGRESSION;
	ifUsesGradientData = true;

}




double GradientKernelRegressionModel::interpolate(rowvec x) const{

	assert(x.size() == dim);

	unsigned int samplesUsedInInterpolation = X.n_rows;

	vec kernelValues(samplesUsedInInterpolation,fill::zeros);

	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		rowvec xi = X.row(i);
		kernelValues(i) = calculateGaussianKernel(x,xi);

	}

	double kernelSum = sum(kernelValues);

	vec weights(samplesUsedInInterpolation,fill::zeros);

	double weightedSum = 0.0;
	for(unsigned int i=0; i<samplesUsedInInterpolation; i++){

		rowvec xi = getRowXRaw(i);
		rowvec xdiff = x-xi;
		rowvec grad = gradientData.row(i);
		double gradTerm = dot(xdiff,grad.row(i));

		weights(i) = kernelValues(i)/kernelSum;
		weightedSum += weights(i)* (y(i)+gradTerm);
	}

	return weightedSum;

}

void GradientKernelRegressionModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{

	cout <<"ERROR: interpolateWithVariance does not exist for the gradient enhanced kernel regression model!\n";
	abort();

}
