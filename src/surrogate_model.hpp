#ifndef SURROAGE_MODEL_HPP
#define SURROAGE_MODEL_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"



using namespace arma;

class SurrogateModel{

protected:
	unsigned int dim;
	unsigned int N;

	mat data;
	mat X;

	std::string label;


	std::string hyperparameters_filename;
	std::string input_filename;

	double ymin,ymax,yave;
	vec xmin;
	vec xmax;
	bool ifUsesGradients;

public:

	SURROGATE_MODEL modelID;

	SurrogateModel();
	SurrogateModel(std::string name, unsigned int dimension);
	virtual void print(void) const;
	virtual void train(void);
	virtual double interpolate(rowvec x) const;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	virtual double calculateInSampleError(void) const;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;

	virtual void validate(mat dataValidation, bool ifVisualize);



};









#endif
