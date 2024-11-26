#include "./INCLUDE/gaussian_correlation_function.hpp"


#include <cassert>
#include <iostream>


namespace Rodop{

void GaussianCorrelationFunction::initialize(void){

	if (dim == 0) {
		throw std::logic_error("Dimension must be set before initialization.");
	}
	vec thetaInit(dim); thetaInit.fill(1.0);
	theta = thetaInit;

}

void GaussianCorrelationFunction::setHyperParameters(const vec &input){

	if (input.isEmpty()) {
		throw std::invalid_argument("Input hyperparameters vector is empty.");
	}
	if (input.getSize() != dim) {
		throw std::invalid_argument("Input hyperparameters vector size does not match the dimension.");
	}
	theta = input ;
}

vec GaussianCorrelationFunction::getHyperParameters(void) const{

	return theta;
}


double GaussianCorrelationFunction::computeCorrelation(const vec &xi, const vec &xj) const {

	return vec::computeGaussianCorrelation(xi.getPointer(), xj.getPointer(),theta.getPointer(),xi.getSize());
}



double GaussianCorrelationFunction::computeCorrelationDot(const vec &xi, const vec &xj, const vec &diffDirection) const {

	return vec::computeGaussianCorrelationDot(xi.getPointer(), xj.getPointer(), diffDirection.getPointer(), theta.getPointer(),xi.getSize());
}




double GaussianCorrelationFunction::computeCorrelationDotDot(const vec &xi, const vec &xj, const vec &firstDiffDirection, const vec &secondDiffDirection) const{


	return vec::computeGaussianCorrelationDotDot(xi.getPointer(),
			xj.getPointer(),
			firstDiffDirection.getPointer(),
			secondDiffDirection.getPointer(),
			theta.getPointer(),
			xi.getSize());


}

void GaussianCorrelationFunction::print(void) const{

	std::cout<<"Theta = \n";
	theta.print();

}

void GaussianCorrelationFunction::computeCorrelationMatrix(void){
	if (!ifInputSampleMatrixIsSet) {
		throw std::logic_error("Input sample matrix is not set.");
	}

	correlationMatrix = X.computeCorrelationMatrixGaussian(theta);
	correlationMatrix.addEpsilonToDiagonal(epsilon);
}
vec GaussianCorrelationFunction::computeCorrelationVector(const vec &x) const{

	if (!ifInputSampleMatrixIsSet) {
		throw std::logic_error("Input sample matrix is not set.");
	}

	vec r(N);
	unsigned int dim = X.getNCols();

	for(unsigned int i=0;i<N;i++){
		vec y = X.getRow(i);
		r(i) = vec::computeGaussianCorrelation(x.getPointer(), y.getPointer(), theta.getPointer(), dim);
	}

	return r;

}


}
