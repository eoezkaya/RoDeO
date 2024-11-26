#include <iostream>
#include <fstream>
#include <cmath>
#include<string>
#include "./INCLUDE/design.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>


using namespace std;

namespace Rodop{


DesignForBayesianOptimization::DesignForBayesianOptimization(){
	dim = 0.0;
	objectiveFunctionValue = 0.0;
	valueAcquisitionFunction = 0.0;
	sigma = 0.0;

};

DesignForBayesianOptimization::DesignForBayesianOptimization(int dimension, int numberOfConstraints){

	dim = dimension;
	objectiveFunctionValue = 0.0;
	valueAcquisitionFunction = 0.0;
	sigma = 0.0;
	constraintValues.resize(numberOfConstraints);
	constraintSigmas.resize(numberOfConstraints);
	dv.resize(dim);
}

DesignForBayesianOptimization::DesignForBayesianOptimization(int dimension){
	dim = dimension;
	objectiveFunctionValue = 0.0;
	valueAcquisitionFunction = 0.0;
	sigma = 0.0;
}

DesignForBayesianOptimization::DesignForBayesianOptimization(const vec& designVector, int numberOfConstraints){

	dv = designVector;
	dim = designVector.getSize();
	constraintValues.resize(numberOfConstraints);
	constraintSigmas.resize(numberOfConstraints);
	objectiveFunctionValue = 0.0;
	valueAcquisitionFunction = 0.0;
	sigma = 0.0;

}

DesignForBayesianOptimization::DesignForBayesianOptimization(const vec& designVector){

	dv = designVector;
	dim = designVector.getSize();
	objectiveFunctionValue = 0.0;
	valueAcquisitionFunction = 0.0;
	sigma = 0.0;

}

void DesignForBayesianOptimization::gradientUpdateDesignVector(const vec &gradient, const vec &lb, const vec &ub, double stepSize){


	/* we go in the direction of gradient since we maximize */
	dv = dv + gradient*stepSize;


	for(unsigned int k=0; k<dim; k++){

		/* if new design vector does not satisfy the box constraints */
		if(dv(k) < lb(k)) dv(k) = lb(k);
		if(dv(k) > ub(k)) dv(k) = ub(k);

	}

}


void DesignForBayesianOptimization::generateRandomDesignVector(void){

	double lowerBound = 0.0;
	double upperBound = 1.0/dim;

	dv.fillRandom(lowerBound, upperBound);

}

void DesignForBayesianOptimization::generateRandomDesignVector(vec lb, vec ub){

	dv.fillRandom(lb,ub);
}

void DesignForBayesianOptimization::generateRandomDesignVectorAroundASample(const vec& sample, const vec& lb, const vec& ub, double factor) {
    if (sample.getSize() != dim) {
        throw std::invalid_argument("Sample dimension mismatch.");
    }

    vec lowerBounds(dim);
    vec upperBounds(dim);

    double dx = factor / dim;

    for (unsigned int i = 0; i < dim; i++) {
        lowerBounds(i) = std::max(sample(i) - dx, lb(i));
        upperBounds(i) = std::min(sample(i) + dx, ub(i));
    }

    dv.fillRandom(lowerBounds, upperBounds);
}


void DesignForBayesianOptimization::print(void) const{
	cout.precision(15);
	cout<<tag<<"n";
	cout<<"Design vector = \n";
	dv.print();
	cout<<"Objective function value = "<<objectiveFunctionValue<<"\n";
	cout<<"Sigma = " << sigma << "\n";
	cout<<"Acquisition function value = "<< valueAcquisitionFunction <<"\n";

	if(constraintValues.getSize() > 0){

		cout<<"Constraint values = \n";
		constraintValues.print();
	}
	if(constraintValues.getSize() > 0){

		cout<<"Constraint feasibility probabilities = \n";
		constraintFeasibilityProbabilities.print();

	}


	cout<<"\n";
}


std::string DesignForBayesianOptimization::toString() const {
    std::ostringstream oss;  // To construct the string with formatting

    // Set precision for floating-point numbers
    oss.precision(15);

    // Append all the data similar to the print function
    oss << "\n" << tag << "\n";
    oss << "Design vector = \n";
    oss << dv.toString() << "\n";

    oss << "Objective function value = " << objectiveFunctionValue << "\n";
    oss << "Sigma = " << sigma << "\n";
    oss << "Acquisition function value = " << valueAcquisitionFunction << "\n";

    if (constraintValues.getSize() > 0) {
        oss << "Constraint values = \n";
        oss << constraintValues.toString() << "\n";
    }

    if (constraintFeasibilityProbabilities.getSize() > 0) {
        oss << "Constraint feasibility probabilities = \n";
        oss << constraintFeasibilityProbabilities.toString() << "\n";
    }

    return oss.str();  // Return the constructed string
}



double  DesignForBayesianOptimization::pdf(double x, double m, double s)
{
	double a = (x - m) / s;

	return INVSQRT2PI / s * std::exp(-0.5 * a * a);
}




double  DesignForBayesianOptimization::cdf(double x0, double mu, double s)
{

	double inp = (x0 - mu) / (s * SQRT2);
	double result = 0.5 * (1.0 + erf(inp));
	return result;
}


double DesignForBayesianOptimization::calculateProbalityThatTheEstimateIsLessThanAValue(double value){

	return cdf(value, objectiveFunctionValue, sigma) ;


}
double DesignForBayesianOptimization::calculateProbalityThatTheEstimateIsGreaterThanAValue(double value){


	return 1.0 - cdf(value, objectiveFunctionValue, sigma) ;

}


void DesignForBayesianOptimization::updateAcqusitionFunctionAccordingToConstraints(void){


	for(unsigned int i=0; i<constraintFeasibilityProbabilities.getSize(); i++){

		valueAcquisitionFunction *= constraintFeasibilityProbabilities(i);
	}


}


void DesignForBayesianOptimization::reset(void){

	constraintFeasibilityProbabilities.fill(0.0);
	constraintSigmas.fill(0.0);
	constraintValues.fill(0.0);
	dv.fill(0.0);
	objectiveFunctionValue  = 0.0;
	sigma  = 0.0;
	valueAcquisitionFunction = 0.0;

}

} /* Namespace Rodop */
