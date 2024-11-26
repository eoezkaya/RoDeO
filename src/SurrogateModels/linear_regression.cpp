#include "./INCLUDE/linear_regression.hpp"
#include "./INCLUDE/surrogate_model.hpp"
#include "../Bounds/INCLUDE/bounds.hpp"

#include <cassert>
#include <stdexcept>
#include <iostream>

namespace Rodop{

//LinearModel::LinearModel():SurrogateModel(){}

void LinearModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);
}


void LinearModel::readData(void){

	assert(!filenameDataInput.empty());
	data.readData(filenameDataInput);
	numberOfSamples = data.getNumberOfSamples();

	ifDataIsRead = true;

}

void LinearModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalize();
	ifNormalized = true;
}


void LinearModel::initializeSurrogateModel(void) {
	// Replace assertions with exception handling
	if (!ifDataIsRead) {
		throw std::runtime_error("Data must be read before initializing the surrogate model.");
	}

	if (dimension <= 0) {
		throw std::invalid_argument("Dimension must be greater than zero to initialize the surrogate model.");
	}

	// Initialization message (could be replaced by a proper logging mechanism)

	if(ifDisplay){
		std::cout << "Initializing the linear model...\n";
	}

	numberOfHyperParameters = dimension + 1;  // Set the number of hyperparameters
	weights.resize(numberOfHyperParameters);  // Resize weights vector
	regularizationParameter = 1e-8;           // Set regularization parameter

	train();  // Train the model

	ifInitialized = true;  // Mark the model as initialized
}

void LinearModel::saveHyperParameters(void) const  {

	assert(!filenameHyperparameters.empty());
	weights.saveToCSV(filenameHyperparameters);

}

void LinearModel::loadHyperParameters(void){

	assert(!filenameHyperparameters.empty());
	weights.readFromCSV(filenameHyperparameters);

}

void LinearModel::printHyperParameters(void) const{

	weights.print("linear regression weights");

}

void LinearModel::setRegularizationParameter(double value){

	regularizationParameter = value;
}

double LinearModel::getRegularizationParameter(void) const{
	return regularizationParameter;
}
vec LinearModel::getWeights(void) const{
	return weights;
}

void LinearModel::setWeights(vec w){
	weights = w;
}

void LinearModel::train(void){

	if (!ifNormalized) {
		throw std::runtime_error("LinearModel::train: Data must be normalized before this operation.");
	}

	if (!ifDataIsRead) {
		throw std::runtime_error("LinearModel::train: Data must be read before this operation.");
	}

	if (dimension == 0) {
		throw std::invalid_argument("LinearModel::train: Dimension must be set before this operation.");
	}

	if (numberOfSamples == 0) {
		throw std::invalid_argument("LinearModel::train: Number of samples must be greater than zero.");
	}

	//	output.printMessage("Finding the weights of the linear model...");


	mat X = data.getInputMatrix();
	vec y = data.getOutputVector();
	mat augmentedX(numberOfSamples, dimension + 1);

	for (unsigned int i = 0; i < numberOfSamples; i++) {

		for (unsigned int j = 0; j <= dimension; j++) {

			if (j == 0){

				augmentedX(i, j) = 1.0;
			}

			else{

				augmentedX(i, j) = X(i, j - 1);
			}
		}
	}

	mat XT = augmentedX.transpose();
	mat XTX = XT *augmentedX;
	XTX.addEpsilonToDiagonal(regularizationParameter);
	mat invXTX = XTX.invert();
	vec XTy = XT.matVecProduct(y);
	weights = invXTX.matVecProduct(XTy);


	/*
	vec res = XTy - XTX.matVecProduct(weights);
	res.print("res", 12);
	 */
	if(ifDisplay){
		weights.print("Linear regression weights");
	}

	ifModelTrainingIsDone = true;


}

double LinearModel::interpolate(vec x ) const{

	double fRegression = weights(0);

	for(unsigned int i=0; i<dimension; i++){

		fRegression += x(i)*weights(i+1);
	}

	return fRegression;

}


void LinearModel::interpolateWithVariance(vec xp,double *fTilde,double *ssqr) const{

	throw std::logic_error("interpolateWithVariance is not valid for the linear model.");
	xp.print();
	*fTilde = 0.0;
	*ssqr = 0.0;

}

double LinearModel::interpolateUsingDerivatives(vec x) const{
	x.print();
	throw std::logic_error("interpolateUsingDerivatives should not be called for SpecificDerivedModel.");

}

vec LinearModel::interpolateAll(mat X) const{

	if (X.getNCols() != dimension) {
		throw std::invalid_argument("LinearModel::interpolateAll: Dimension of the matrix does not match with problem dimension.");
	}

	unsigned int N = X.getNRows();

	vec result(N);

	for(unsigned int i=0; i<N; i++){
		result(i) = interpolate(X.getRow(i));
	}

	return result;
}


void LinearModel::printSurrogateModel(void) const{

	data.print();
	weights.print("regression weights");

}


void LinearModel::addNewSampleToData(vec newsample){
	throw std::logic_error("No need to use this function within the Linear model!");
	newsample.print();

}

void LinearModel::addNewLowFidelitySampleToData(vec newsample){
	throw std::logic_error("No need to use this function within the Linear model!");
	newsample.print();
}


void LinearModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;

}

void LinearModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}


void LinearModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());
	filenameHyperparameters = filename;

}

void LinearModel::updateModelWithNewData(void){



}


} /* Namespace Rodop */
