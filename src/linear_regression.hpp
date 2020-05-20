#ifndef TRAIN_LINREG_HPP
#define TRAIN_LINREG_HPP
#include <armadillo>
using namespace arma;
#include "surrogate_model.hpp"

void train_linear_regression(mat &X, vec &ys, vec &w, double lambda);

class LinearModel : public SurrogateModel {

	vec weights;
	double regularizationParam;

public:

	LinearModel();
	LinearModel(std::string name, unsigned int dimension);
	void print(void) const;
	void train(void);
	double interpolate(rowvec x) const;
	vec interpolateAll(mat X) const;

	void setRegularizationParam(double value);
	double getRegularizationParam(void) const;
	double calculateInSampleError(void) const;

};


#endif
