/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
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

#include "exponential_correlation_function.hpp"
#include <cassert>
#include "matrix_vector_operations.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;




void ExponentialCorrelationFunction::setTheta(vec input){

	assert(dim>0);
	assert(input.empty() == false);
	assert(input.size() == dim);

	thetaParameters = input;

}

void ExponentialCorrelationFunction::setGamma(vec input){

	assert(dim>0);
	assert(input.empty() == false);
	assert(input.size() == dim);

	gammaParameters = input;
}

void ExponentialCorrelationFunction::setHyperParameters(vec input){

	assert(input.empty() == false);
	assert(dim>0);

	setTheta(input.head(dim));
	setGamma(input.tail(dim));

}

vec ExponentialCorrelationFunction::getHyperParameters(void) const{

	assert(dim>0);
	vec hyperParameters(2*dim, fill::zeros);

	for(unsigned int i=0; i<dim; i++){

		hyperParameters(i) = thetaParameters(i);

	}

	for(unsigned int i=0; i<dim; i++){

		hyperParameters(i+dim) = gammaParameters(i);

	}

	return hyperParameters;
}



bool ExponentialCorrelationFunction::checkIfParametersAreSetProperly(void) const{

	if(gammaParameters.size() != dim) {

		std::cout<<"gamma size = "<<gammaParameters.size() <<"\n";
		return false;
	}
	if(thetaParameters.size() != dim) {

		std::cout<<"theta size = "<<thetaParameters.size() <<"\n";
		return false;
	}

	if(gammaParameters.max() > 2.0) {

		std::cout<<"gamma max = "<<gammaParameters.max() <<"\n";

		return false;
	}
	if(gammaParameters.min() < 0.0) {

		std::cout<<"gamma min = "<<gammaParameters.min() <<"\n";
		return false;
	}

	if(thetaParameters.min() < 0.0) {

		std::cout<<"theta min = "<<thetaParameters.min() <<"\n";

		return false;
	}

	return true;
}


double ExponentialCorrelationFunction::computeCorrelation(const rowvec &xi, const rowvec &xj) const {

	double sum = 0.0;

	for (unsigned int k = 0; k < dim; k++) {

		double exponentialPart = pow(fabs(xi(k) - xj(k)), gammaParameters(k));
		sum += thetaParameters(k) * exponentialPart;
	}

	double correlation = exp(-sum);
	return correlation;
}


double ExponentialCorrelationFunction::computeCorrelationDot(const rowvec &xi, const rowvec &xj, const rowvec &direction) const {
    double sum  = 0.0;
    double sumd = 0.0;
    double fabs0 = 0.0;
    double fabs0d = 0.0;

    for (int k = 0; k < dim; ++k) {
        if (xi(k) - xj(k) >= 0.0) {
            fabs0d = -direction(k);
            fabs0 = xi(k) - xj(k);
        } else {
            fabs0d = direction(k);
            fabs0 = -(xi(k)-xj(k));
        }
        double exponentialPart  = pow(fabs0, gammaParameters(k));
        double exponentialPartd = gammaParameters(k)*pow(fabs0, (gammaParameters(k)-1))*fabs0d;

        sumd = sumd + thetaParameters(k)*exponentialPartd;
        sum += thetaParameters(k)*exponentialPart;
    }
    double correlationd;
    correlationd = -(exp(-sum)*sumd);

    return correlationd;
}


double ExponentialCorrelationFunction::computeCorrelationDotDot(const rowvec &xi,
		const rowvec &xj, const rowvec &direction2, const rowvec &direction1){
    double sum = 0.0;
    double sumd0;
    double sumd = 0.0;
    double sumdd;
    double fabs0 = 0.0;
    double fabs0d0;
    double fabs0d = 0.0;
    double temp;
    sumd0 = 0.0;
    sumdd = 0.0;
    for (int k = 0; k < dim; ++k) {
        double tempd;
        double temp;
        if (xi(k) - xj(k) >= 0.0) {
            fabs0d = -direction1(k);
            fabs0d0 = -direction2(k);
            fabs0 = xi(k) - xj(k);
        } else {
            fabs0d = direction1(k);
            fabs0d0 = direction2(k);
            fabs0 = -(xi(k)-xj(k));
        }
        double exponentialPart = pow(fabs0, gammaParameters(k));
        double exponentialPartd0 = (fabs0 <= 0.0 && (gammaParameters(k) == 0.0 || gammaParameters(k) != (int)gammaParameters(k)) ? 0.0 : gammaParameters(k)*pow(fabs0, (gammaParameters(k)-1))*fabs0d0
        );
        temp = gammaParameters(k) - 1;
        tempd = (fabs0 <= 0.0 && (temp == 0.0 || temp != (int)temp) ? 0.0 :
            temp*pow(fabs0, (temp-1))*fabs0d0);
        double exponentialPartd = gammaParameters(k)*fabs0d*pow(fabs0, temp);
        double exponentialPartdd = gammaParameters(k)*fabs0d*tempd;
        sumdd = sumdd + thetaParameters(k)*exponentialPartdd;
        sumd = sumd + thetaParameters(k)*exponentialPartd;
        sumd0 = sumd0 + thetaParameters(k)*exponentialPartd0;
        sum += thetaParameters(k)*exponentialPart;
    }
    temp = exp(-sum);
    double correlationdd = -(temp*sumdd-sumd*exp(-sum)*sumd0);
    return correlationdd;
}






void ExponentialCorrelationFunction::initialize(void){

	assert(dim>0);
	vec thetaInit(dim); thetaInit.fill(1.0);
	vec gammaInit(dim); gammaInit.fill(2.0);

	setTheta(thetaInit);
	setGamma(gammaInit);


}

void ExponentialCorrelationFunction::print(void) const{

	std::cout<<"Exponential correlation function = \n";
	thetaParameters.print("theta:");
	gammaParameters.print("theta:");
	printScalar(epsilon);



}
