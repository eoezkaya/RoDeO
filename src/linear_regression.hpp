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
	LinearModel(std::string name);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolate(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	double calculateInSampleError(void) const;

	vec interpolateAll(mat X) const;

	void setRegularizationParam(double value);
	double getRegularizationParam(void) const;

};


#endif
