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

	bool ifInitialized;


public:

	SURROGATE_MODEL modelID;

	SurrogateModel();
	SurrogateModel(std::string name, unsigned int dimension);


	void ReadDataAndNormalize(void);

	virtual void initializeSurrogateModel(void);
	virtual void printSurrogateModel(void) const;
	virtual void printHyperParameters(void) const;
	virtual void train(void);
	virtual double interpolate(rowvec x) const;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	virtual double calculateInSampleError(void) const;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;

	mat tryModelOnTestSet(mat testSet) const;
	void visualizeTestResults(mat testResults) const;



};









#endif
