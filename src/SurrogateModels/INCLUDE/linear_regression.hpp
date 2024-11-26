#ifndef TRAIN_LINREG_HPP
#define TRAIN_LINREG_HPP
#include "surrogate_model.hpp"


namespace Rodop{

//void train_linear_regression(mat &X, vec &ys, vec &w, double lambda);

class LinearModel : public SurrogateModel {

	vec weights;
	double regularizationParameter;

public:

	//	LinearModel();
	//	LinearModel(std::string name);

	void setBoxConstraints(Bounds boxConstraintsInput);

	void readData(void);
	void normalizeData(void);

	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);
	void setNumberOfTrainingIterations(unsigned int);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolate(vec x) const ;

	double interpolateUsingDerivatives(vec x) const;
	double interpolateWithGradients(vec x) const ;
	void interpolateWithVariance(vec xp,double *f_tilde,double *ssqr) const;

	void addNewSampleToData(vec newsample);
	void addNewLowFidelitySampleToData(vec newsample);

	vec interpolateAll(mat X) const;

	void setRegularizationParameter(double value);
	double getRegularizationParameter(void) const;
	vec getWeights(void) const;
	void setWeights(vec);

	void updateModelWithNewData(void);

};


} /* Namespace Rodop */

#endif
