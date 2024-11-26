#include "./INCLUDE/correlation_functions.hpp"
#include <stdexcept>

namespace Rodop{

CorrelationFunctionBase::CorrelationFunctionBase(){}



bool CorrelationFunctionBase::isInputSampleMatrixSet(void) const{

	return ifInputSampleMatrixIsSet;

}

void CorrelationFunctionBase::setInputSampleMatrix(const mat& input){

	if (input.isEmpty()) {
		throw std::invalid_argument("Input sample matrix cannot be empty.");
	}
	X = input;
	N = X.getNRows();
	dim = X.getNCols();
	correlationMatrix.reset();
	correlationMatrix.resize(N,N);
	ifInputSampleMatrixIsSet = true;


}

void CorrelationFunctionBase::setDimension(unsigned int input){
	if (input <= 0) {
		throw std::invalid_argument("Dimension must be positive.");
	}
	dim = input;
}

void CorrelationFunctionBase::setEpsilon(double value){

	if (value < 0.0 || value >= 1.0) {
		throw std::out_of_range("Epsilon must be in the range [0.0, 1.0).");
	}
	epsilon = value;

}

Rodop::mat  CorrelationFunctionBase::getCorrelationMatrix(void) const{

	return correlationMatrix;

}

}




