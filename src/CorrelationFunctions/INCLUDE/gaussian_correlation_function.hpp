#ifndef GAUSSIAN_CORRELATION_FUNCTION_HPP
#define GAUSSIAN_CORRELATION_FUNCTION_HPP

#include "correlation_functions.hpp"


namespace Rodop{

class GaussianCorrelationFunction : public CorrelationFunctionBase{

private:


	vec theta;


public:

	void initialize(void);

	void setHyperParameters(const vec&);
	vec getHyperParameters(void) const;


	double computeCorrelationDotDot(const vec &, const vec &, const vec &, const vec &) const;

	double computeCorrelation(const vec &xi, const vec &xj) const override;
	double computeCorrelationDot(const vec &xi, const vec &xj, const vec &xdot) const override;

	void computeCorrelationMatrix(void) override;
	vec computeCorrelationVector(const vec &x) const override;



	void print(void) const;


};

}

#endif
