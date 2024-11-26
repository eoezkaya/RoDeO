#include "./INCLUDE/surrogate_model_data.hpp"
#include "./INCLUDE/model_logger.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>


namespace Rodop{

SurrogateModelData::SurrogateModelData(){}


void SurrogateModelData::reset(void){

	X.reset();
	XTest.reset();
	Xraw.reset();
	XrawTest.reset();
	dimension = 0;
	gradient.reset();
	differentiationDirections.reset();
	directionalDerivatives.reset();
	ifDataHasGradients = false;
	ifDataIsNormalized = false;
	ifDataIsRead = false;
	numberOfSamples = 0;
	numberOfTestSamples = 0;
	boxConstraints.reset();

}

unsigned int SurrogateModelData::getNumberOfSamples(void) const{
	return numberOfSamples;
}
unsigned int SurrogateModelData::getNumberOfSamplesValidation(void) const{
	return numberOfValidationSamples;
}

unsigned int SurrogateModelData::getNumberOfSamplesTest(void) const{
	return numberOfTestSamples;
}


unsigned int SurrogateModelData::getDimension(void) const{
	return dimension;
}


void SurrogateModelData::setDimension(unsigned int value){
	dimension = value;
}


bool SurrogateModelData::isDataRead(void) const{
	return ifDataIsRead;
}

mat SurrogateModelData::getRawData(void) const{
	return rawData;
}


void SurrogateModelData::setGradientsOn(void){
	ifDataHasGradients = true;
}

void SurrogateModelData::setGradientsOff(void){
	ifDataHasGradients = false;
}

void SurrogateModelData::setDirectionalDerivativesOn(void){
	ifDataHasDirectionalDerivatives = true;
}

void SurrogateModelData::setDirectionalDerivativesOff(void){
	ifDataHasDirectionalDerivatives = false;
}


std::vector<int> SurrogateModelData::generateRandomIndicesForValidationData(int N, double ratio) const {
	// Calculate k based on the ratio
	int k = static_cast<int>(ratio * N);

	// Ensure k is within the valid range
	if (k > N) k = N;
	if (k < 0) k = 0;

	// Create a vector to store the indices
	std::vector<int> indices(N);
	std::vector<int> selectedIndices;

	// Initialize the vector with sequential indices from 0 to N-1
	for (int i = 0; i < N; ++i) {
		indices[i] = i;
	}

	// Use random device and Mersenne Twister for randomness
	std::random_device rd;
	std::mt19937 gen(rd());

	// Shuffle the indices
	std::shuffle(indices.begin(), indices.end(), gen);

	// Select the first k indices from the shuffled vector
	selectedIndices.insert(selectedIndices.end(), indices.begin(), indices.begin() + k);

	return selectedIndices;
}




void SurrogateModelData::readData(string inputFilename){

	assert(!inputFilename.empty());
	assert(!(ifDataHasDirectionalDerivatives == true && ifDataHasGradients == true));

	printMessage("Loading data from the file: " + inputFilename + "\n");

	rawData.readFromCSV(inputFilename);
	rawDataWork = rawData;

	if(ifDisplay){
		rawData.print("rawData");
	}

	ModelLogger::getInstance().log(INFO, "Ratio for validation samples: " + std::to_string(ratioValidationData) );

	if(ratioValidationData>0){
		numberOfSamples = rawData.getNRows();
		indicesForValidationData = generateRandomIndicesForValidationData(numberOfSamples, ratioValidationData);

		for (int idx : indicesForValidationData) {
			vec validationSample = rawData.getRow(idx);
			rawDataValidation.addRow(validationSample,-1);
		}

		numberOfValidationSamples = indicesForValidationData.size();
		rawDataWork.deleteRows(indicesForValidationData);
	}

	numberOfSamples = rawDataWork.getNRows();
	ModelLogger::getInstance().log(INFO, "Number of training samples: " + std::to_string(numberOfSamples) );
	ModelLogger::getInstance().log(INFO, "Number of validation samples: " + std::to_string(numberOfValidationSamples) );


	if(ifDataHasGradients){
		printMessage("Data has gradients... \n");
	}

	if(ifDataHasDirectionalDerivatives){
		printMessage("Data has tangents... \n");
	}

	assignDimensionFromData();

	assignSampleInputMatrix();
	assignSampleOutputVector();

	assignGradientMatrix();

	assignDirectionalDerivativesVector();
	assignDifferentiationDirectionMatrix();

	filenameFromWhichTrainingDataIsRead = inputFilename;
	ifDataIsRead = true;


}


void SurrogateModelData::revertBackToFullData() {

	ModelLogger::getInstance().log(INFO, "SurrogateModelData: Reverting back to use full data...");

	numberOfValidationSamples = 0;
	rawDataWork = rawData;
	numberOfSamples = rawData.getNRows();

	ModelLogger::getInstance().log(INFO, "Number of training samples: " + std::to_string(numberOfSamples) );
	ModelLogger::getInstance().log(INFO, "Number of validation samples: " + std::to_string(numberOfValidationSamples) );

	yValidation.reset();
	XValidation.reset();
	XrawValidation.reset();


	assignSampleInputMatrix();
	assignSampleOutputVector();

	assignGradientMatrix();

	assignDirectionalDerivativesVector();
	assignDifferentiationDirectionMatrix();

	normalize();

}


void SurrogateModelData::readDataTest(string inputFilename) {

	// Check if the input filename is not empty
	if (inputFilename.empty()) {
		throw std::invalid_argument("Input filename is empty.");
	}

	// Check if the dimension is greater than zero
	if (dimension == 0) {
		throw std::runtime_error("Invalid dimension: Dimension must be greater than zero.");
	}

	printMessage("Loading test data from the file: " + inputFilename + "\n");

	// Read data from the CSV file
	XrawTest.readFromCSV(inputFilename);

	// Get the number of test samples and print it
	numberOfTestSamples = XrawTest.getNRows();
	//   output.printMessage("Number of test samples = ", numberOfTestSamples);

	// Check the number of columns in the test data
	if (XrawTest.getNCols() == dimension + 1) {
		// Data contains function values
		yTest = XrawTest.getCol(dimension);  // Extract the last column for function values
		XTest = XrawTest.submat(0,numberOfTestSamples - 1,  0, dimension - 1);  // Extract input data
		ifTestDataHasFunctionValues = true;
	} else if (XrawTest.getNCols() == dimension) {
		// Data contains only input values
		XTest = XrawTest;
	} else {
		// Invalid number of columns
		throw std::runtime_error("Problem with test data (CSV format): incorrect number of columns in file: " + inputFilename);
	}

	// Mark test data as read
	ifTestDataIsRead = true;
}



void SurrogateModelData::assignDimensionFromData(void) {

	if(rawDataWork.isEmpty()){
		throw std::runtime_error("Error: RawDataWork is empty (assign dimension from data).");
	}
	unsigned int dimensionOfTrainingData;

	// Determine the dimension based on the type of data
	if (ifDataHasGradients) {
		dimensionOfTrainingData = (rawDataWork.getNCols() - 1) / 2;
	} else if (ifDataHasDirectionalDerivatives) {
		dimensionOfTrainingData = (rawDataWork.getNCols() - 2) / 2;
	} else {
		dimensionOfTrainingData = rawDataWork.getNCols() - 1;
	}

	// Check if the dimension is already set and if it matches the derived dimension
	if (dimension > 0 && dimensionOfTrainingData != dimension) {
		std::cerr << "dimension = " <<dimension<<"\n";
		std::cerr << "dimension of the training data = " << dimensionOfTrainingData << "\n";
		throw std::runtime_error("Error: Dimension of the training data does not match the specified dimension (assign dimension from data).");
	}

	// Assign the calculated dimension to the member variable
	dimension = dimensionOfTrainingData;

	// Print a message indicating the identified dimension
	printMessage("Dimension of the problem is identified as " + std::to_string(dimension));
}

void SurrogateModelData::checkDimensionAndNumberOfSamples() {
	if (dimension == 0) {
		throw std::runtime_error("Dimension must be greater than 0.");
	}
	if (numberOfSamples == 0) {
		throw std::runtime_error("Number of samples must be greater than 0.");
	}
}

void SurrogateModelData::assignSampleInputMatrix(void){

	ModelLogger::getInstance().log(INFO,"SurrogateModelData: assign sample input matrix...");
	checkDimensionAndNumberOfSamples();

	X = rawDataWork.submat(0,numberOfSamples-1,0, dimension-1);

	if(numberOfValidationSamples>0){
		ModelLogger::getInstance().log(INFO,"SurrogateModelData: assign sample input matrix for validation samples...");

		XValidation = rawDataValidation.submat(0, numberOfValidationSamples-1, 0, dimension-1);


		if (XValidation.getNRows() != numberOfValidationSamples) {
			throw std::runtime_error("The number of rows in sample input matrix does not match number of validation samples.");
		}
		if (XValidation.getNCols() != dimension) {
			throw std::runtime_error("The number of columns in sample input matrix does not match dimension.");
		}
		XrawValidation = XValidation;
	}

	/* we save the raw data */
	Xraw = X;


	// Ensure the integrity of the sample input matrix
	if (X.getNRows() != numberOfSamples) {
		throw std::runtime_error("The number of rows in sample input matrix does not match numberOfSamples.");
	}
	if (X.getNCols() != dimension) {
		throw std::runtime_error("The number of columns in sample input matrix does not match dimension.");
	}

}

void SurrogateModelData::assignSampleOutputVector(void){
	ModelLogger::getInstance().log(INFO,"SurrogateModelData: assign sample output vector...");
	checkDimensionAndNumberOfSamples();
	y = rawDataWork.getCol(dimension);

	if(numberOfValidationSamples>0){
		ModelLogger::getInstance().log(INFO,"SurrogateModelData: assign sample output vector for validation samples...");
		yValidation = rawDataValidation.getCol(dimension);
	}


}

void SurrogateModelData::assignGradientMatrix(void){

	checkDimensionAndNumberOfSamples();


	if(ifDataHasGradients){

		if(rawDataWork.getNCols() <= dimension+1){
			throw std::runtime_error("The number of columns does not match when data has gradients.");

		}

		gradient = rawDataWork.submat(0, numberOfSamples - 1, dimension+1, 2*dimension);

	}


}


void SurrogateModelData::assignDirectionalDerivativesVector(void){

	assert(dimension>0);

	if(ifDataHasDirectionalDerivatives){
		directionalDerivatives = rawDataWork.getCol(dimension+1);
	}
}


void SurrogateModelData::assignDifferentiationDirectionMatrix(void){

	checkDimensionAndNumberOfSamples();

	if(ifDataHasDirectionalDerivatives){
		differentiationDirections = rawDataWork.submat(0, numberOfSamples-1, dimension+2, 2*dimension+1);
	}
}

void SurrogateModelData::normalize(void){

	assert(ifDataIsRead);
	assert(boxConstraints.areBoundsSet());

	normalizeSampleInputMatrix();
	normalizeDerivativesMatrix();


	if(ifDataHasGradients){
		normalizeGradientMatrix();
	}

	ifDataIsNormalized = true;
}


void SurrogateModelData::normalizeSampleInputMatrix(void){

	ModelLogger::getInstance().log(INFO,"SurrogateModelData: normalize sample input matrix...");
	if (X.getNRows() != numberOfSamples) {
		throw std::runtime_error("Mismatch between the number of rows in X and the number of samples.");
	}
	if (X.getNCols() != dimension) {
		throw std::runtime_error("Mismatch between the number of columns in X and the expected dimension.");
	}
	if (boxConstraints.getDimension() != dimension) {
		throw std::runtime_error("Box constraints' dimension does not match the input matrix dimension.");
	}
	if (!boxConstraints.areBoundsSet()) {
		throw std::runtime_error("Box constraints are not set.");
	}


	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;

	if (deltax.has_zeros()) {
		throw std::runtime_error("Zero range found in box constraints, leading to potential division by zero.");
	}

	X = X.normalizeMatrix(xmin,xmax);

	if(numberOfValidationSamples>0){
		ModelLogger::getInstance().log(INFO,"SurrogateModelData: normalize sample input matrix for validation samples...");
		XValidation = XValidation.normalizeMatrix(xmin,xmax);
	}

	ifDataIsNormalized = true;
}


void SurrogateModelData::normalizeSampleInputMatrixTest(void){

	if (!boxConstraints.areBoundsSet()) {
		throw std::runtime_error("Box constraints are not set.");
	}

	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;
	if (deltax.has_zeros()) {
		throw std::runtime_error("Zero range found in box constraints, leading to potential division by zero.");
	}

	XTest = XTest.normalizeMatrix(xmin,xmax);

}

void SurrogateModelData::normalizeDerivativesMatrix(void){

	if(ifDataHasDirectionalDerivatives){

		assert(ifDataIsRead);
		assert(boxConstraints.areBoundsSet());

		vec lb = boxConstraints.getLowerBounds();
		vec ub = boxConstraints.getUpperBounds();
		/* This factor is assumed to be same for all variables */
		double scalingFactor = (ub(0) - lb(0))*dimension;
		assert(scalingFactor > 0.0);
		directionalDerivatives =  directionalDerivatives*scalingFactor;

	}

}


void SurrogateModelData::normalizeGradientMatrix(void){

	assert(ifDataIsRead);
	assert(boxConstraints.areBoundsSet());
	assert(dimension>0);
	assert(gradient.getNCols() == dimension);


	gradientRaw = gradient;

	for(unsigned int i=0; i<dimension; i++){

		vec lb = boxConstraints.getLowerBounds();
		vec ub = boxConstraints.getUpperBounds();
		double scalingFactor = (ub(i) - lb(i))*dimension;
		assert(scalingFactor > 0.0);
		vec scaledGradient = gradient.getCol(i)*scalingFactor;
		gradient.setCol(scaledGradient,i);

	}


}

vec SurrogateModelData::getRowGradient(unsigned int index) const{

	return gradient.getRow(index);

}

vec SurrogateModelData::getRowGradientRaw(unsigned int index) const{

	return gradientRaw.getRow(index);

}

vec SurrogateModelData::getDirectionalDerivativesVector(void) const{

	return directionalDerivatives;


}


vec SurrogateModelData::getRowDifferentiationDirection(unsigned int index) const{

	return differentiationDirections.getRow(index);

}


vec SurrogateModelData::getRowRawData(unsigned int index) const{

	return rawData.getRow(index);

}



vec SurrogateModelData::getRowX(unsigned int index) const{

	assert(index < X.getNRows());

	return X.getRow(index);

}

vec SurrogateModelData::getRowXTest(unsigned int index) const{

	assert(index < XTest.getNRows());

	return XTest.getRow(index);

}




vec SurrogateModelData::getRowXRaw(unsigned int index) const{

	assert(index < Xraw.getNRows());

	return Xraw.getRow(index);
}

vec SurrogateModelData::getRowXRawTest(unsigned int index) const{

	assert(index < XrawTest.getNRows());

	return XrawTest.getRow(index);
}


vec SurrogateModelData::getOutputVector(void) const{

	return y;

}

vec SurrogateModelData::getOutputVectorValidation(void) const{

	return yValidation;

}


vec SurrogateModelData::getOutputVectorTest(void) const{

	assert(ifTestDataIsRead);
	return XrawTest.getCol(dimension);

}


void SurrogateModelData::setValidationRatio(double val){
	if(val<0.0 || val>1.0){
		throw std::invalid_argument("SurrogateModelData: Ratio of validation samples must be between 0 and 1.");
	}
	ratioValidationData = val;

}


void SurrogateModelData::setOutputVector(vec yIn){

	assert(yIn.getSize() == numberOfSamples);
	y = yIn;


}


mat SurrogateModelData::getInputMatrix(void) const{
	return X;
}

mat SurrogateModelData::getInputMatrixTest(void) const{
	return XTest;
}

mat SurrogateModelData::getInputMatrixValidation(void) const{
	return XValidation;
}




double SurrogateModelData::getMinimumOutputVector(void) const{
	return y.findMin();
}

double SurrogateModelData::getMaximumOutputVector(void) const{
	return y.findMax();
}

mat SurrogateModelData::getGradientMatrix(void) const{
	return gradient;
}

void SurrogateModelData::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;

}



Bounds SurrogateModelData::getBoxConstraints(void) const{

	return boxConstraints;

}

bool SurrogateModelData::isDataNormalized(void) const{
	return ifDataIsNormalized;
}


void SurrogateModelData::printHere(const std::string& file , int line) {
	std::cout << file << " " << line << "\n";
}


void SurrogateModelData::printValidationSampleIndices(void) const{

	std::cout<<"Indices of validation samples = \n";
	for (int val : indicesForValidationData) {
		std::cout << val << " ";
	}
	std::cout << std::endl;

}

void SurrogateModelData::print(void) const{

	rawData.print("raw data");
	X.print("X");
	y.print("sample output vector");


	if(numberOfValidationSamples>0){
		printValidationSampleIndices();
		XValidation.print("X validation");
		yValidation.print("Output vector for validation samples");
	}


	if(ifDataHasGradients){

		gradient.print("sample gradient matrix");

	}

	if(ifDataHasDirectionalDerivatives){

		directionalDerivatives.print("directional derivatives");
		differentiationDirections.print("differentiation directions");

	}

	if(ifTestDataIsRead){

		XrawTest.print("raw data for testing");
		XTest.print("sample input matrix for testing");

	}

}

void SurrogateModelData::printMessage(string msg){
	if(ifDisplay) std::cout<<msg<<"\n";
}



} /*Namespace Rodop */


