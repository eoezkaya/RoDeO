#ifndef SURROGATE_MODEL_TESTER_HPP
#define SURROGATE_MODEL_TESTER_HPP


#include <string>
#include "../../Bounds/INCLUDE/bounds.hpp"

#include "./linear_regression.hpp"
#include "./kriging_training.hpp"


namespace Rodop{


class SurrogateModelTester{


private:


	std::string name;

	unsigned int dimension = 0;
	Bounds boxConstraints;

	LinearModel linearModel;
	KrigingModel krigingModel;


	SurrogateModel *surrogateModel;

	SURROGATE_MODEL surrogateModelType;

	unsigned int numberOfTrainingIterations = 10000;

	std::string fileNameTraingData;
	std::string fileNameTraingDataLowFidelity;
	std::string fileNameTestData;

	void checkBoxConstraints();
	void checkIfSurrogateModelIsSpecified();
	void checkFilename(const std::string& filename);

public:

	SurrogateModelTester();

	void setName(std::string);

	void setDimension(unsigned int);

	void setNumberOfTrainingIterations(unsigned int);
	void setRatioValidationSamples(double value);

	void setSurrogateModel(SURROGATE_MODEL);
	void bindSurrogateModels(void);

	void setBoxConstraints(Bounds);

	void performSurrogateModelTest(void);

	void setFileNameTrainingData(std::string);


	void setFileNameTestData(std::string);

	bool ifReadWarmStart = false;

	bool ifSurrogateModelSpecified = false;
	bool ifSurrogateModelLowFiSpecified = false;
	bool ifbindSurrogateModelisDone = false;
	bool ifMultiLevel = false;

	bool ifVariancesShouldBeComputedInTest = false;

	void print(void) const;
};

} /* Namespace Rodop */

#endif
