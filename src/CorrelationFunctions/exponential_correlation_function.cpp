#include <cassert>
#include <iostream>

#include "./INCLUDE/exponential_correlation_function.hpp"

#include <limits>
#include <cmath>

using namespace std;

namespace Rodop{



void ExponentialCorrelationFunction::setTheta(const vec& input){

	if (input.isEmpty() || input.getSize() != dim) {
		throw std::invalid_argument("Invalid theta parameters.");
	}

	thetaParameters = input;

}

void ExponentialCorrelationFunction::setGamma(const vec& input){

	if (input.isEmpty() || input.getSize() != dim) {
		throw std::invalid_argument("Invalid gamma parameters.");
	}

	gammaParameters = input;
}

void ExponentialCorrelationFunction::setHyperParameters(const vec &input){

	if (input.getSize()!= 2*dim) {
		throw std::invalid_argument("Invalid hyperparameters input.");
	}


	setTheta(input.head(dim));
	setGamma(input.tail(dim));

}

vec ExponentialCorrelationFunction::getHyperParameters(void) const{

	if (dim == 0) {
		throw std::logic_error("Dimension must be set before getting hyperparameters.");
	}
	vec hyperParameters(2*dim);

	for(unsigned int i=0; i<dim; i++){

		hyperParameters(i) = thetaParameters(i);

	}

	for(unsigned int i=0; i<dim; i++){

		hyperParameters(i+dim) = gammaParameters(i);

	}

	return hyperParameters;
}



bool ExponentialCorrelationFunction::checkIfParametersAreSetProperly(void) const {
    if (gammaParameters.getSize() != dim) {
        std::cerr << "Invalid gamma size: " << gammaParameters.getSize() << "\n";
        return false;
    }
    if (thetaParameters.getSize() != dim) {
        std::cerr << "Invalid theta size: " << thetaParameters.getSize() << "\n";
        return false;
    }
    if (gammaParameters.findMax() > 2.0 || gammaParameters.findMin() < 0.0) {
        std::cerr << "Gamma parameters out of range.\n";
        return false;
    }
    if (thetaParameters.findMin() < 0.0) {
        std::cerr << "Theta parameters out of range.\n";
        return false;
    }
    return true;
}


double ExponentialCorrelationFunction::computeCorrelation(const vec &xi, const vec &xj) const {

	return vec::computeExponentialCorrelation(xi.getPointer(),
			xj.getPointer(),
			thetaParameters.getPointer(),
			gammaParameters.getPointer(),
			xi.getSize());
}


double ExponentialCorrelationFunction::computeCorrelationDot(const vec &xi, const vec &xj, const vec &direction) const {

	return vec::computeExponentialCorrelationDot(xi.getPointer(),
			xj.getPointer(),
			direction.getPointer(),
			thetaParameters.getPointer(),
			gammaParameters.getPointer(),
			xi.getSize());
}


void ExponentialCorrelationFunction::initialize(void) {
    if (dim == 0) {
        throw std::logic_error("Dimension must be set before initialization.");
    }
    vec thetaInit(dim);
    thetaInit.fill(1.0);
    vec gammaInit(dim);
    gammaInit.fill(1.8);

    setTheta(thetaInit);
    setGamma(gammaInit);
}

void ExponentialCorrelationFunction::print(void) const {
    std::cout << "Exponential correlation function:\n";
    thetaParameters.print("Theta:");
    gammaParameters.print("Gamma:");

}


void ExponentialCorrelationFunction::computeCorrelationMatrix(void){

	if(!ifInputSampleMatrixIsSet){
		throw std::invalid_argument("Input matrix is not set.");
	}
	correlationMatrix = X.computeCorrelationMatrixExponentialNaive(thetaParameters, gammaParameters);

	/* It seems this is absolutely necessary */
	correlationMatrix.addEpsilonToDiagonal(epsilon);


}
vec ExponentialCorrelationFunction::computeCorrelationVector(const vec &x) const{

	if(!ifInputSampleMatrixIsSet){
		throw std::invalid_argument("Input matrix is not set.");
	}
	vec r(N);
	int dim = X.getNCols();

	for(unsigned int i=0;i<N;i++){
		vec y = X.getRow(i);
		r(i) = vec::computeExponentialCorrelation(x.getPointer(), y.getPointer(), thetaParameters.getPointer(), gammaParameters.getPointer(), dim);
	}

	return r;

}

}
