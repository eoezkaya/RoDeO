
#include "./INCLUDE/surrogate_model_tester.hpp"
#include "./INCLUDE/model_logger.hpp"
#include <cassert>


namespace Rodop{


SurrogateModelTester::SurrogateModelTester(){}


void SurrogateModelTester::setDimension(unsigned int value){
	dimension = value;
}

void SurrogateModelTester::setName(string nameInput){
	assert(!nameInput.empty());

	name = nameInput;
}

void SurrogateModelTester::setNumberOfTrainingIterations(unsigned int nIterations) {
	numberOfTrainingIterations = nIterations;
	if(ifbindSurrogateModelisDone){
		surrogateModel->setNumberOfTrainingIterations(numberOfTrainingIterations);

	}
}

void SurrogateModelTester::setSurrogateModel(SURROGATE_MODEL modelType){

	surrogateModelType = modelType;
	ifSurrogateModelSpecified = true;

}

void SurrogateModelTester::setSurrogateModelLowFi(SURROGATE_MODEL modelType){

	surrogateModelTypeLowFi = modelType;
	ifSurrogateModelLowFiSpecified = true;

}

void SurrogateModelTester::setRatioValidationSamples(double value){

	if(value<0 && value > 1.0){
		ModelLogger::getInstance().log(ERROR, "Model Tester: Validation samples ration is not valid.");
		throw std::invalid_argument("Model Tester: Validation samples ration is not valid.");
	}
	surrogateModel->setRatioValidationSamples(value);
}

void SurrogateModelTester::checkIfSurrogateModelIsSpecified() {
	if (!ifSurrogateModelSpecified) {
		ModelLogger::getInstance().log(ERROR,
				"Model Tester: Surrogate model must be set first.");
		throw std::runtime_error(
				"Model Tester: Surrogate model must be set first.");
	}
}

void SurrogateModelTester::bindSurrogateModels(void){

	checkIfSurrogateModelIsSpecified();

	if(!ifMultiLevel){

		if(surrogateModelType == LINEAR_REGRESSION ){
			ModelLogger::getInstance().log(INFO, "Model Tester: Linear model...");
			surrogateModel = &linearModel;
		}
		if(surrogateModelType == ORDINARY_KRIGING){
			ModelLogger::getInstance().log(INFO, "Model Tester: Ordinary Kriging...");
			surrogateModel = &krigingModel;
		}
		if(surrogateModelType == UNIVERSAL_KRIGING){
			ModelLogger::getInstance().log(INFO, "Model Tester: Universal Kriging...");
			krigingModel.setLinearRegressionOn();
			surrogateModel = &krigingModel;
		}


	}

	else{
		ModelLogger::getInstance().log(INFO, "Model Tester: Multi-Fidelity feature is active...");

		assert(ifSurrogateModelLowFiSpecified);
		multilevelModel.setIDHiFiModel(surrogateModelType);
		multilevelModel.setIDLowFiModel(surrogateModelTypeLowFi);
		multilevelModel.setinputFileNameHighFidelityData(fileNameTraingData);
		multilevelModel.setinputFileNameLowFidelityData(fileNameTraingDataLowFidelity);
		multilevelModel.bindModels();

		surrogateModel = &multilevelModel;
	}

	ifbindSurrogateModelisDone = true;

}

void SurrogateModelTester::setBoxConstraints(Bounds boxConstraintsInput){
	boxConstraints = boxConstraintsInput;
	checkBoxConstraints();
}

void SurrogateModelTester::checkBoxConstraints() {
	if (!boxConstraints.areBoundsSet()) {
		ModelLogger::getInstance().log(ERROR,
				"Model Tester: Box constraints must be set first.");
		throw std::runtime_error(
				"Model Tester: Box constraints must be set first.");
	}
}

void SurrogateModelTester::performSurrogateModelTest(void){

	checkBoxConstraints();

	assert(ifbindSurrogateModelisDone);
	assert(!name.empty());
	assert(dimension>0);
	checkFilename(fileNameTraingData);
	checkFilename(fileNameTestData);

	ModelLogger::getInstance().log(INFO, "#################################.");
	ModelLogger::getInstance().log(INFO, "Model Tester: Perform model test.");


	surrogateModel->setDimension(dimension);
	surrogateModel->setName(name);
	surrogateModel->setNameOfInputFile(fileNameTraingData);
	surrogateModel->setNameOfInputFileTest(fileNameTestData);

	surrogateModel->setBoxConstraints(boxConstraints);

	ModelLogger::getInstance().log(INFO, "Model Tester: Reading training data.");
	surrogateModel->readData();
	ModelLogger::getInstance().log(INFO, "Model Tester: Reading test data.");
	surrogateModel->readDataTest();

	surrogateModel->setBoxConstraints(boxConstraints);
	//	outputToScreen.printMessage("Data normalization...");
	surrogateModel->normalizeData();
	surrogateModel->normalizeDataTest();
	ModelLogger::getInstance().log(INFO, "Model Tester: Initializing surrogate model.");
	surrogateModel->initializeSurrogateModel();
	surrogateModel->setNumberOfTrainingIterations(numberOfTrainingIterations);

	if(ifReadWarmStart){
		surrogateModel->setReadWarmStartFileFlag(true);
	}
	ModelLogger::getInstance().log(INFO, "Model Tester: Training surrogate model.");
	surrogateModel->train();

	if(ifVariancesShouldBeComputedInTest){
		ModelLogger::getInstance().log(INFO, "Model Tester: Evaluation of test data estimates with variances.");
		surrogateModel->tryOnTestData("with_variances");
		surrogateModel->saveTestResultsWithVariance();
	}
	else{
		ModelLogger::getInstance().log(INFO, "Model Tester: Evaluation of test data estimates.");
		surrogateModel->tryOnTestData();
		surrogateModel->saveTestResults();

	}


	surrogateModel->printGeneralizationError();

	double inSampleError = surrogateModel->calculateInSampleError();
	ModelLogger::getInstance().log(INFO,"Training error (MSE): " + std::to_string(inSampleError ));


}

void SurrogateModelTester::checkFilename(const std::string& filename) {
    if (filename.empty()) {
        ModelLogger::getInstance().log(ERROR, "Model Tester: Empty filename.");
        throw std::invalid_argument("Model Tester: Empty filename.");
    }

    // Check for invalid characters
    if (filename.find_first_of("\\/:*?\"<>|") != std::string::npos) {
        ModelLogger::getInstance().log(ERROR, "Model Tester: Filename contains invalid characters.");
        throw std::invalid_argument("Model Tester: Filename contains invalid characters.");
    }

    // Check for CSV extension
    std::string extension = ".csv";
    if (filename.size() < extension.size() ||
        filename.substr(filename.size() - extension.size()) != extension) {
        ModelLogger::getInstance().log(ERROR, "Model Tester: Filename must end with .csv extension.");
        throw std::invalid_argument("Model Tester: Filename must end with .csv extension.");
    }

}

void SurrogateModelTester::setFileNameTrainingData(string filename){

	checkFilename(filename);
	fileNameTraingData = filename;

}


void SurrogateModelTester::setFileNameTrainingDataLowFidelity(string filename){

	checkFilename(filename);
	fileNameTraingDataLowFidelity = filename;

}

void SurrogateModelTester::setFileNameTestData(string filename){

	checkFilename(filename);
	fileNameTestData = filename;

}

void SurrogateModelTester::print(void) const{}

} /* Namespace Rodop */


