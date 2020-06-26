#ifndef GRADIENT_KERNEL_REG_HPP
#define GRADIENT_KERNEL_REG_HPP

#include "Rodeo_macros.hpp"
#include "kernel_regression.hpp"

class GradientKernelRegressionModel: public KernelRegressionModel{

private:


public:

	GradientKernelRegressionModel();
	GradientKernelRegressionModel(std::string name);

	double interpolate(rowvec x) const;
	void interpolateWithVariance(rowvec,double *,double *) const;


};



#endif
