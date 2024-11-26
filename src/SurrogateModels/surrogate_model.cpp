#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>
#include<cmath>
#include "./INCLUDE/surrogate_model.hpp"
#include "./INCLUDE/model_logger.hpp"


namespace Rodop{


SurrogateModel::SurrogateModel(){}

void SurrogateModel::setName(std::string nameInput){

	assert(!nameInput.empty());

	name = nameInput;

	filenameHyperparameters = name + "_hyperparameters.csv";

}

string SurrogateModel::getName(void) const{
	return name;
}


void SurrogateModel::setDimension(unsigned int dim){

	dimension = dim;
}


void SurrogateModel::setNumberOfThreads(unsigned int n){

	numberOfThreads = n;

}


void SurrogateModel::setGradientsOn(void){

	data.setGradientsOn();
	ifHasGradientData = true;


}

void SurrogateModel::setGradientsOff(void){

	data.setGradientsOff();
	ifHasGradientData = false;

}

bool SurrogateModel::areGradientsOn(void) const{

	return ifHasGradientData;

}

void SurrogateModel::setWriteWarmStartFileFlag(bool flag){

	ifWriteWarmStartFile = flag;
}

void SurrogateModel::setReadWarmStartFileFlag(bool flag){
	ifReadWarmStartFile = flag;

}


void SurrogateModel::setRatioValidationSamples(double val){
	data.setValidationRatio(val);
	ifHasValidationSamples = true;
}


std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return filenameHyperparameters;

}


std::string SurrogateModel::getNameOfInputFile(void) const{

	return filenameDataInput;

}


unsigned int SurrogateModel::getDimension(void) const{
	return dimension;
}
unsigned int SurrogateModel::getNumberOfSamples(void) const{
	return numberOfSamples;
}

mat SurrogateModel::getRawData(void) const{
	return data.getRawData();
}

mat SurrogateModel::getX(void) const{

	return data.getInputMatrix();
}

SurrogateModelData SurrogateModel::getData() const{
	return data;
}

vec SurrogateModel::gety(void) const{

	return data.getOutputVector();

}


void SurrogateModel::printData(void) const{
	data.print();
}

void SurrogateModel::readDataTest(void){

	assert(!filenameDataInputTest.empty());

	data.readDataTest(filenameDataInputTest);

	ifTestDataIsRead = true;
}

unsigned int SurrogateModel::countHowManySamplesAreWithinBounds(vec lb, vec ub){

	assert(ifDataIsRead);

	mat trainingData;
	trainingData = data.getRawData();

	unsigned int N   = data.getNumberOfSamples();
	unsigned int dim = data.getDimension();

	unsigned int counter = 0;
	for(unsigned int i=0; i<N; i++){

		vec sample     = trainingData.getRow(i);
		vec dv         = sample.head(dim);

		if( dv.is_between(lb,ub) ){
			counter++;
		}

	}

	return counter;
}


void SurrogateModel::normalizeDataTest(void){

	data.normalizeSampleInputMatrixTest();
	ifNormalizedTestData = true;

}

void SurrogateModel::updateAuxilliaryFields(void){

}


vec SurrogateModel::interpolateVector(mat X) const{


	unsigned int N = X.getNRows();
	vec results(N);

	for(unsigned int i=0; i<N; i++){

		vec xp = X.getRow(i);
		results(i) = interpolate(xp);

	}

	return results;
}


double SurrogateModel::calculateInSampleError(string mode) const{

	if(!ifInitialized){

		ModelLogger::getInstance().log(ERROR, "Surrogate model: Model must be initialized first.");
		throw std::runtime_error("Surrogate model: Model must be initialized first.");
	}

	ModelLogger::getInstance().log(INFO, "Surrogate model: Calculating in sample error using " + std::to_string(data.getNumberOfSamples()) + " samples");

	vec fTildeValues(data.getNumberOfSamples());
	mat rawData = data.getRawData();
	vec fExact = rawData.getCol(dimension);

	mat X = data.getInputMatrix();
	double squaredError = 0.0;
	for(unsigned int i=0; i<data.getNumberOfSamples(); i++){
		vec x = X.getRow(i);
		ModelLogger::getInstance().log(INFO, "x = " +  x.toString());

		if(mode == "function_values"){
			fTildeValues(i) = interpolate(x);
		}
		else if(mode == "with_derivatives"){

			fTildeValues(i) = interpolateUsingDerivatives(x);
		}
		else{
			string msg = "SurrogateModel::calculateInSampleError: invalid mode";
			ModelLogger::getInstance().log(ERROR, msg);
			throw std::runtime_error(msg);

		}


		std::string msg = "value: " + std::to_string(fExact(i)) + " estimate:" + std::to_string(fTildeValues(i));
		ModelLogger::getInstance().log(INFO,msg);
		squaredError += (fTildeValues(i) - fExact(i))*(fTildeValues(i) - fExact(i));

	}

	return squaredError/data.getNumberOfSamples();

}



double SurrogateModel::calculateOutSampleError(void){

	if(!ifHasTestData){
		ModelLogger::getInstance().log(ERROR, "Surrogate model: Test data does not exist.");
		throw std::runtime_error("Surrogate model: Test data does not exist.");
	}
	if(!data.ifTestDataHasFunctionValues){
		ModelLogger::getInstance().log(ERROR, "Surrogate model: Test data does have functional values.");
		throw std::runtime_error("Surrogate model: Test data does have functional values.");
	}

	tryOnTestData();

	vec squaredErrors = testResults.getCol(data.getDimension()+2);

	return squaredErrors.mean();

}

void SurrogateModel::writeFileHeaderForTestResults() {

	unsigned int dim = data.getDimension();
	for (unsigned int i = 0; i < dim; i++) {
		string variableName = "x" + std::to_string(i + 1);
		testResultsFileHeader.push_back(variableName);
	}
	testResultsFileHeader.push_back("Estimated value");
	if (data.ifTestDataHasFunctionValues) {
		testResultsFileHeader.push_back("True value");
		testResultsFileHeader.push_back("Squared Error");
	}
}

void SurrogateModel::saveTestResults(void){

	if(filenameTestResults.empty()){
		ModelLogger::getInstance().log(ERROR, "Surrogate model: Filename for test results is empty.");
		throw std::runtime_error("Surrogate model: Filename for test results is empty.");
	}

	writeFileHeaderForTestResults();
	testResults.saveAsCSV(filenameTestResults,6, testResultsFileHeader);
	ModelLogger::getInstance().log(INFO, "Surrogate model: Writing results to the file = " +  filenameTestResults);

}


void SurrogateModel::saveTestResultsWithVariance(void){

	if(filenameTestResults.empty()){
		ModelLogger::getInstance().log(ERROR, "Surrogate model: Filename for test results is empty.");
		throw std::runtime_error("Surrogate model: Filename for test results is empty.");
	}

	writeFileHeaderForTestResults();

	testResultsFileHeader.push_back("Variance");

	testResults.saveAsCSV(filenameTestResults,6, testResultsFileHeader);

}


void SurrogateModel::printSurrogateModel(void) const{
	data.print();
}
vec SurrogateModel::getRowX(unsigned int index) const{
	return data.getRowX(index);
}

vec SurrogateModel::getRowXRaw(unsigned int index) const{
	return data.getRowXRaw(index);
}

void SurrogateModel::setNameOfInputFileTest(string filename){

	assert(!filename.empty());
	filenameDataInputTest = filename;

	ifHasTestData = true;
}

void SurrogateModel::setNameOfOutputFileTest(string filename){

	assert(!filename.empty());
	filenameTestResults = filename;
}

double SurrogateModel::calculateValidationError(void) {
	unsigned int N = data.getNumberOfSamplesValidation();

	if (N > 0) {
		if (!ifNormalized) {
			throw std::runtime_error("Validation data must be normalized first. Please call the normalization method before calculating the validation error.");
		}

		mat XValidation = data.getInputMatrixValidation();
		vec yValidation = data.getOutputVectorValidation();  // Corrected method to get the output vector

		double error = 0.0;
		for (unsigned int i = 0; i < N; i++) {
			vec x = XValidation.getRow(i);
			double ftilde = interpolate(x);

			//            x.print("x");
			//            std::cout<< "ftilde = " << ftilde << "\n";
			//            std::cout<< "f      = " << yValidation(i) << "\n";
			//            std::cout<< "\n\n";

			//            error += (ftilde - yValidation(i)) * (ftilde - yValidation(i));
			error += fabs(ftilde - yValidation(i));
		}

		error /= static_cast<double>(N); // Calculate mean squared error (MSE)
		//       ModelLogger::getInstance().log(INFO, "Validation error (MSE) = " + std::to_string(error));

		return error;
	} else {
		ModelLogger::getInstance().log(WARNING, "calculateValidationError did nothing, because there is no validation data.");
		return std::numeric_limits<double>::max();  // Return the largest possible double value
	}
}



void SurrogateModel::tryOnTestData(string mode){

	if(!ifNormalizedTestData){
		throw std::runtime_error("Test data must be normalized first. Please call the normalization method before calculating the test error.");
	}

	unsigned int dim = data.getDimension();
	unsigned int numberOfEntries;

	if(data.ifTestDataHasFunctionValues){

		numberOfEntries = dim + 3;
		if(mode == "with_variances") numberOfEntries = dim + 4;

	}
	else{

		numberOfEntries = dim + 1;
		if(mode == "with_variances") numberOfEntries = dim + 2;
	}

	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();
	vec squaredError(numberOfTestSamples);


	mat results(numberOfTestSamples,numberOfEntries);

	vec fExact;
	if(data.ifTestDataHasFunctionValues){
		fExact = data.getOutputVectorTest();
	}

	mat XTest = data.getInputMatrixTest();


	ModelLogger::getInstance().log(INFO, "Surrogate model: Calculating test error using " + std::to_string(numberOfTestSamples) + " samples");


	for(unsigned int i=0; i<numberOfTestSamples; i++){

		vec xp          = data.getRowXTest(i);
		vec dataRow     = data.getRowXRawTest(i);
		vec x = dataRow.head(dimension);

		double fTilde = 0.0;
		double variance = 0.0;
		if(mode == "function_values"){

			fTilde = interpolate(xp);
		}
		else if(mode == "with_derivatives"){

			fTilde = interpolateUsingDerivatives(xp);
		}
		else if(mode == "with_variances"){

			interpolateWithVariance(xp, &fTilde, &variance);
		}
		else{
			string errMsg = "SurrogateModel::tryOnTestData: invalid mode.";
			ModelLogger::getInstance().log(ERROR,errMsg);
			throw std::runtime_error(errMsg);

		}

		vec sample = x;
		sample.push_back(fTilde);

		if(data.ifTestDataHasFunctionValues){
			sample.push_back(fExact(i));
			double error = pow((fExact(i) - fTilde),2.0);
			squaredError(i) = error;
			sample.push_back(error);
			ModelLogger::getInstance().log(INFO,"x = " + x.toString());
			std::string msg = "value: " + std::to_string(fExact(i)) + " estimate:" + std::to_string(fTilde) + " error = " + std::to_string(error);
			ModelLogger::getInstance().log(INFO,msg);
//			x.print("x");
//			std::cout<<msg<<"\n\n";


		}
		else{

			ModelLogger::getInstance().log(INFO,"x = " + x.toString());
			std::string msg = "estimate:" + std::to_string(fTilde);
		}
		if(mode == "with_variances"){
			ModelLogger::getInstance().log(INFO,"variance = " + std::to_string(variance));
			ModelLogger::getInstance().log(INFO,"standard deviation = " + std::to_string(sqrt(variance)));

			sample.push_back(variance);
		}

		results.setRow(sample,i);
	}

	if(data.ifTestDataHasFunctionValues){

		generalizationError = squaredError.mean();
		standardDeviationOfGeneralizationError = squaredError.standardDeviation();
	}


	testResults = results;

	ModelLogger::getInstance().log(INFO,"Test error (MSE): " + std::to_string(generalizationError));

}


void SurrogateModel::printGeneralizationError(void) const{
	std::cout << std::fixed << std::setprecision(8);
	std::cout<<"Generalization error (MSE) : " << generalizationError << "\n";

}



} /*Namespace Rodop */
