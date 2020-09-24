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


void GradientKernelRegressionModel::printHyperParameters(void) const{
	cout << "Here\n";
	printMatrix(mahalanobisMatrix,"mahalanobisMatrix");
	cout << "sigma = " <<sigmaGaussianKernel << "\n";
}

void GradientKernelRegressionModel::printHyperParameters2(void) const{
	cout << "Here\n";
	printMatrix(mahalanobisMatrix,"mahalanobisMatrix");
	cout << "sigma = " <<sigmaGaussianKernel << "\n";
}


double GradientKernelRegressionModel::interpolateWithGradients(rowvec x) const{

	cout << "Here\n";
	abort();

	assert(x.size() == dim);

	rowvec xp = normalizeRowVectorBack(x, xmin, xmax);

	printVector(x,"x");
	printVector(xp,"xp");

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

		printVector(xi,"xi");

		rowvec xdiff = xp-xi;

		printVector(xdiff,"xdiff");
		rowvec grad = gradientData.row(i);

		printVector(grad,"grad");
		double gradTerm = dot(xdiff,grad.row(i));

		weights(i) = kernelValues(i)/kernelSum;
		weightedSum += weights(i)* (y(i)+gradTerm);
	}

	return weightedSum;

}
