#ifndef SURROAGE_MODEL_HPP
#define SURROAGE_MODEL_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"



using namespace arma;


class PartitionData{

public:

	std::string label;
	mat rawData;
	mat X;
	vec yExact;
	vec ySurrogate;
	vec squaredError;

	PartitionData();
	PartitionData(std::string name);
	void fillWithData(mat);
	bool ifNormalized;
	unsigned int numberOfSamples;
	unsigned int dim;
	void normalizeAndScaleData(vec xmin, vec xmax);
	double calculateMeanSquaredError(void) const;
	rowvec getRow(unsigned int indx) const;
	void saveAsCSVFile(std::string fileName);
	void print(void) const;

};

class SurrogateModel{

protected:
	unsigned int dim;
	unsigned int N;

	mat rawData;
	mat X;
	mat gradientData;
	vec y;


	std::string label;
	std::string hyperparameters_filename;
	std::string input_filename;


	double ymin,ymax,yave;
	vec xmin;
	vec xmax;

	bool ifInitialized;
	bool ifUsesGradientData;


public:

	SURROGATE_MODEL modelID;

	SurrogateModel();
	SurrogateModel(std::string name);


	void ReadDataAndNormalize(void);

	virtual void initializeSurrogateModel(void) = 0;
	virtual void printSurrogateModel(void) const = 0;
	virtual void printHyperParameters(void) const = 0;
	virtual void saveHyperParameters(void) const = 0;
	virtual void loadHyperParameters(void) = 0;
	virtual void train(void) = 0;
	virtual double interpolate(rowvec x) const = 0;
	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const = 0;

	virtual double calculateInSampleError(void) const = 0;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;


	void tryModelOnTestSet(PartitionData &testSet) const;
	void visualizeTestResults(void) const;



};









#endif
