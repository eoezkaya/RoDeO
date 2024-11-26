#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<cassert>
#include <limits>
#include <iomanip>
#include <thread>
#include "./INCLUDE/kriging_training.hpp"
#include "./INCLUDE/linear_regression.hpp"
#include "./INCLUDE/model_logger.hpp"


#ifdef OPENMP_SUPPORT
#include <omp.h>
#endif

namespace Rodop{

KrigingModel::KrigingModel():SurrogateModel(){}

void KrigingModel::setDimension(unsigned int dim){

	dimension = dim;
	linearModel.setDimension(dim);
	data.setDimension(dim);
}

void KrigingModel::setNameOfInputFile(std::string filename) {

	if (filename.empty()) {
		ModelLogger::getInstance().log(ERROR, "KrigingModel::setNameOfInputFile: Filename cannot be empty. Please provide a valid filename.");
		throw std::invalid_argument("Filename cannot be empty. Please provide a valid filename.");
	}

	filenameDataInput = filename;
	linearModel.setNameOfInputFile(filename);
}


void KrigingModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void KrigingModel::setNameOfHyperParametersFile(std::string filename){

	if (filename.empty()) {
		throw std::invalid_argument("Filename cannot be empty. Please provide a valid filename.");
	}
	filenameHyperparameters = filename;

}


void KrigingModel::setBoxConstraints(Bounds boxConstraintsInput) {
	if (!boxConstraintsInput.areBoundsSet()) {
		throw std::invalid_argument("KrigingModel::setBoxConstraints: Box constraints are not set. Please provide valid constraints.");
	}
	if (dimension == 0) {
		throw std::runtime_error("KrigingModel::setBoxConstraints: Dimension must be set first.");
	}

	if(boxConstraintsInput.getDimension() != dimension){
		throw std::runtime_error("KrigingModel::setBoxConstraints: Dimensions do not match.");
	}

	// If using linear regression, set the box constraints for the linear model
	if (ifUsesLinearRegression) {
		linearModel.setBoxConstraints(boxConstraintsInput);
	}

	// Set the box constraints for the Kriging model
	boxConstraints = boxConstraintsInput;
	data.setBoxConstraints(boxConstraintsInput);

}

void KrigingModel::setObjectiveFunctionTypeForModelTraining(KrigingObjectiveType type){
	objectiveFunctionType = type;
}


void KrigingModel::readData(void) {

	checkDimension();

	// Check if the input data filename is set
	if (filenameDataInput.empty()) {
		throw std::runtime_error("The filename for input data is not set.");
	}

	// Read data using the specified filename
	try {
		ModelLogger::getInstance().log(INFO, "Kriging model: Reading training data from file: " + filenameDataInput);

		data.readData(filenameDataInput);
	} catch (const std::exception& e) {
		throw std::runtime_error("Failed to read input data: " + std::string(e.what()));
	}

	// If a linear regression model is being used, read its data as well
	if (ifUsesLinearRegression) {
		try {
			linearModel.readData();
		} catch (const std::exception& e) {
			throw std::runtime_error("Failed to read data for linear regression model: " + std::string(e.what()));
		}
	}

	// Update the number of samples and set the flag indicating data has been read
	numberOfSamples = data.getNumberOfSamples();
	ifDataIsRead = true;
}


void KrigingModel::normalizeData(void){
	checkIfDataIsRead();

	ModelLogger::getInstance().log(INFO, "KrigingModel::normalizeData...");

	data.normalize();
	if(ifUsesLinearRegression){
		ModelLogger::getInstance().log(INFO, "KrigingModel::normalizeData for the linear model");
		linearModel.normalizeData();
	}
	ifNormalized = true;
}

void KrigingModel::checkIfDataIsRead() {
	if (!ifDataIsRead) {
		ModelLogger::getInstance().log(ERROR, "KrigingModel: Data has not been read.");
		throw std::runtime_error(
				"KrigingModel::initializeSurrogateModel: Data has not been read.");
	}
}

void KrigingModel::initializeSurrogateModel(void){

	checkDimension();

	if (numberOfSamples == 0) {
		throw std::invalid_argument("KrigingModel::initializeSurrogateModel: Number of samples must be greater than zero.");
	}
	checkIfDataIsRead();

	if (!ifNormalized) {
		throw std::runtime_error("KrigingModel::initializeSurrogateModel: Data has not been normalized.");
	}

	ModelLogger::getInstance().log(INFO, "KrigingModel::initializeSurrogateModel...");

	mat X = data.getInputMatrix();
	correlationFunction.setInputSampleMatrix(X);

	if(!ifCorrelationFunctionIsInitialized){
		ModelLogger::getInstance().log(INFO, "KrigingModel::initializeSurrogateModel: Initializing the correlation function...");
		correlationFunction.initialize();
		ifCorrelationFunctionIsInitialized = true;
	}


	numberOfHyperParameters = 2*dimension;


	if(ifUsesLinearRegression){
		ModelLogger::getInstance().log(INFO, "KrigingModel::initializeSurrogateModel: Linear model is active");

		linearModel.initializeSurrogateModel();
		vec ys = data.getOutputVector();
		mat X = data.getInputMatrix();
		vec ysLinearRegression = linearModel.interpolateAll(X);
		ys = ys - ysLinearRegression;

		data.setOutputVector(ys);

	}

	updateAuxilliaryFields();

	ModelLogger::getInstance().log(INFO, "Kriging model: Initialization is done ...");
	ifInitialized = true;
}


void KrigingModel::calculateBeta0(void) {

	vec ys = data.getOutputVector();
	double sumys = ys.sum();
	beta0 = sumys / numberOfSamples;
}

void KrigingModel::checkDimension() {
	if (dimension == 0) {
		ModelLogger::getInstance().log(ERROR, "KrigingModel: Invalid dimension detected. The dimension must be set.");
		throw std::runtime_error(
				"KrigingModel: Dimension must be set first.");
	}
}

void KrigingModel::updateAuxilliaryFields(void){


	checkIfDataIsRead();

	if (!ifNormalized) {
		throw std::runtime_error("KrigingModel::updateAuxilliaryFields: Data has not been normalized.");
	}
	if (!ifCorrelationFunctionIsInitialized) {
		throw std::runtime_error("KrigingModel::updateAuxilliaryFields: Correlation function is not initialized.");
	}
	if (numberOfSamples == 0) {
		throw std::invalid_argument("KrigingModel::updateAuxilliaryFields: Number of training samples must be greater than zero.");
	}
	checkDimension();

	correlationFunction.computeCorrelationMatrix();
	mat R = correlationFunction.getCorrelationMatrix();

	linearSystemCorrelationMatrix.setMatrix(R);

	/* Cholesky decomposition R = L L^T */

	linearSystemCorrelationMatrix.factorize();


	resetDataObjects();
	resizeDataObjects();

	if(linearSystemCorrelationMatrix.isFactorizationDone()){

		/* solve R x = ys */
		vec ys = data.getOutputVector();

		R_inv_ys = linearSystemCorrelationMatrix.solveLinearSystem(ys);

		/* solve R x = I */

		R_inv_I = linearSystemCorrelationMatrix.solveLinearSystem(vectorOfOnes);

		beta0 = (1.0/ (vectorOfOnes.dot(R_inv_I))* (vectorOfOnes.dot(R_inv_ys)));

		if(ifUseAverageBeta0){
			beta0 = ys.mean();
		}

		vec ys_min_betaI = ys - beta0;

		/* solve R x = ys-beta0*I */

		R_inv_ys_min_beta = linearSystemCorrelationMatrix.solveLinearSystem( ys_min_betaI);

		sigmaSquared = (1.0 / numberOfSamples) * ys_min_betaI.dot(R_inv_ys_min_beta);

	}

}


void KrigingModel::printHyperParameters(void) const{

	unsigned int dim = data.getDimension();

	std::cout<<"Hyperparameters of the Kriging model = \n";


	vec hyperParameters = correlationFunction.getHyperParameters();
	vec theta = hyperParameters.head(dim);
	vec gamma = hyperParameters.tail(dim);

	theta.print("theta");
	gamma.print("gamma");

	if(ifUsesLinearRegression){

		vec w = linearModel.getWeights();
		std::cout<<"Weights of the linear regression model = \n";
		w.print();

	}

}

void KrigingModel::logHyperParameters() const {
	unsigned int dim = data.getDimension();

	// Header message for hyperparameters
	ModelLogger::getInstance().log(INFO, "Hyperparameters of the Kriging model:");

	// Get hyperparameters (theta and gamma) from the correlation function
	vec hyperParameters = correlationFunction.getHyperParameters();
	vec theta = hyperParameters.head(dim);
	vec gamma = hyperParameters.tail(dim);

	// Convert theta and gamma to strings and log them
	ModelLogger::getInstance().log(INFO, "Theta (correlation length scale parameters): " + theta.toString() );
	ModelLogger::getInstance().log(INFO, "Gamma (smoothness parameters): " + gamma.toString());


	// If linear regression is used, log the weights
	if (ifUsesLinearRegression) {
		vec w = linearModel.getWeights();
		ModelLogger::getInstance().log(INFO, "Weights of the linear regression model: " + w.toString() );
	}

	ModelLogger::getInstance().log(INFO, "End of hyperparameters log.\n");
}


void KrigingModel::setHyperParameters(vec parameters){

	if(parameters.getSize() != numberOfHyperParameters){
		ModelLogger::getInstance().log(ERROR, "KrigingModel: size of the perameter vector does not match with number of hyperparameters.");
		throw std::invalid_argument("KrigingModel: size of the perameter vector does not match with number of hyperparameters.");
	}
	correlationFunction.setHyperParameters(parameters);


}

vec KrigingModel::getHyperParameters(void) const{

	return correlationFunction.getHyperParameters();
}

void KrigingModel::checkFilenameHyperparameters() const {
	if (filenameHyperparameters.empty()) {
		ModelLogger::getInstance().log(ERROR,
				"Filename for hyperparameters must be specified.");
		throw std::invalid_argument(
				"Filename for hyperparameters must be specified.");
	}
}

void KrigingModel::saveHyperParameters(void) const{

	checkFilenameHyperparameters();

	ModelLogger::getInstance().log(INFO, "Saving hyperparameters into the file: " + filenameHyperparameters);

	vec saveBuffer = correlationFunction.getHyperParameters();
	saveBuffer.saveToCSV(filenameHyperparameters);

}

void KrigingModel::loadHyperParameters(void){

	checkDimension();
	checkFilenameHyperparameters();

	vec loadBuffer;
	loadBuffer.readFromCSV(filenameHyperparameters);


	unsigned int numberOfEntriesInTheBuffer = loadBuffer.getSize();


	if(numberOfEntriesInTheBuffer == numberOfHyperParameters) {

		unsigned int dim = data.getDimension();

		vec theta = loadBuffer.head(dim);
		vec gamma = loadBuffer.tail(dim);

		ModelLogger::getInstance().log(INFO, "Loading hyperparameters from the file: " + filenameHyperparameters);
		ModelLogger::getInstance().log(INFO, "Theta: " + theta.toString());
		ModelLogger::getInstance().log(INFO, "Gamma: " + gamma.toString());

		correlationFunction.setTheta(theta);
		correlationFunction.setGamma(gamma);

	}
	else{

		ModelLogger::getInstance().log(WARNING, "Cannot load hyperparameters from the file: " + filenameHyperparameters);

	}

}


vec KrigingModel::getRegressionWeights(void) const{
	return linearModel.getWeights();
}
void KrigingModel::setRegressionWeights(vec weights){
	linearModel.setWeights(weights);
}

void KrigingModel::setEpsilon(double value){

	if (value < 0) {
		ModelLogger::getInstance().log(ERROR,
				"Regularization parameter cannot be nagative.");
		throw std::invalid_argument(
				"Regularization parameter cannot be nagative.");
	}

	correlationFunction.setEpsilon(value);

}

void KrigingModel::setLinearRegressionOn(void){

	ifUsesLinearRegression  = true;

}
void KrigingModel::setLinearRegressionOff(void){

	ifUsesLinearRegression  = false;

}


void KrigingModel::addNewSampleToData(vec newsample) {
	if (newsample.getSize() != data.getDimension() + 1) {
		throw std::runtime_error("KrigingModel::addNewSampleToData: Sample size does not match expected dimension.");
	}

	ModelLogger::getInstance().log(INFO, "KrigingModel: Adding new sample to the training data");
	ModelLogger::getInstance().log(INFO, "sample = " + newsample.toString());


	// Avoid points that are too close to each other
	mat rawData = data.getRawData();

	int rowIndex = rawData.findRowIndex(newsample, 1e-4);

	if (rowIndex == -1) {
		try {
			newsample.appendToCSV(filenameDataInput,12);  // Append new sample to the CSV file
			updateModelWithNewData();  // Update the model with the new data
		} catch (const std::exception& e) {
			// Log the error if writing to CSV or updating the model fails
			ModelLogger::getInstance().log(ERROR, std::string("Failed to add new sample: ") + e.what());
		}
	} else {
		// Log a warning if the new sample is too close to an existing one
		ModelLogger::getInstance().log(WARNING, "The new sample is too close to an existing sample and has been discarded.");
	}
}


void KrigingModel::addNewLowFidelitySampleToData(vec newsample){

	newsample.print();
	throw std::logic_error("No need to use this function within the Kriging model!");

}


void KrigingModel::printSurrogateModel(void) const {
	// Setting precision for floating-point numbers
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "\nKriging Surrogate Model:\n";
	std::cout << "-------------------------------------\n";
	std::cout << "Number of samples: " << data.getNumberOfSamples() << std::endl;
	std::cout << "Number of input parameters: " << data.getDimension() << std::endl;
	std::cout << "Hyperparameters file name: " << (filenameHyperparameters.empty() ? "Not set" : filenameHyperparameters) << std::endl;
	std::cout << "Training data file name: " << (filenameDataInput.empty() ? "Not set" : filenameDataInput) << std::endl;
	std::cout << "Maximum number of iterations for inner optimization: " << numberOfTrainingIterations << std::endl;
	std::cout << "\n";

	std::cout << "Data: \n";
	if (data.ifDataIsRead) {
		data.print();
	} else {
		std::cout << "Data has not been read yet.\n";
	}


	if(boxConstraints.areBoundsSet()){
		std::cout << "\nBox constraints: \n";
		boxConstraints.print();
	}

	if (ifCorrelationFunctionIsInitialized) {
		std::cout << "\nCorrelation Function: \n";
		correlationFunction.print();

		std::cout << "\nHyperparameters: \n";
		printHyperParameters();
	} else {
		std::cout << "Correlation function is not initialized.\n";
	}

	std::cout << "beta0 = " << beta0 << "\n";

	std::cout << "-------------------------------------\n";
}


void KrigingModel::resetDataObjects(void){
	R_inv_ys.reset();
	R_inv_I.reset();
	R_inv_ys_min_beta.reset();
	vectorOfOnes.reset();
	beta0 = 0.0;
	sigmaSquared = 0.0;
}

void KrigingModel::resizeDataObjects(void){

	if (numberOfSamples == 0) {
		throw std::logic_error("KrigingModel::resizeDataObjects: number of samples cannot be zero.");
	}

	R_inv_ys_min_beta.resize(numberOfSamples);
	R_inv_I.resize(numberOfSamples);
	R_inv_ys.resize(numberOfSamples);
	vectorOfOnes.resize(numberOfSamples);
	vectorOfOnes.fill(1.0);

}



void KrigingModel::checkAuxilliaryFields(void) const{

	mat R = correlationFunction.getCorrelationMatrix();
	vec ys = data.getOutputVector();
	vec ys_min_betaI = ys - vectorOfOnes*beta0;
	vec residual2 = ys_min_betaI - R.matVecProduct(R_inv_ys_min_beta);

	residual2.print("residual2", 9);
	//	printVector(residual2,"residual (ys-betaI - R * R^-1 (ys-beta0I) )");
	vec residual3 = vectorOfOnes - R.matVecProduct(R_inv_I);
	residual3.print("residual3", 9);
	//	printVector(residual3,"residual (I - R * R^-1 I)");
}

void KrigingModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();
	normalizeData();
	initializeSurrogateModel();

}

double KrigingModel::interpolate(vec xp ) const{


	double estimateLinearRegression = 0.0;
	double estimateKriging = 0.0;

	if(ifUsesLinearRegression ){

		estimateLinearRegression = linearModel.interpolate(xp);
	}

	vec r = correlationFunction.computeCorrelationVector(xp);

	estimateKriging = beta0 + r.dot(R_inv_ys_min_beta);

	return estimateLinearRegression + estimateKriging;

}

double KrigingModel::interpolateUsingDerivatives(vec x ) const{
	throw std::logic_error("No need to use this function within the Kriging model!");
	x.print();
}




void KrigingModel::interpolateWithVariance(vec xp,double *ftildeOutput,double *sSqrOutput) const{

	unsigned int N = data.getNumberOfSamples();
	*ftildeOutput =  interpolate(xp);


	vec R_inv_r(N);

	vec r = correlationFunction.computeCorrelationVector(xp);

	/* solve the linear system R x = r by Cholesky matrices U and L*/

	R_inv_r = linearSystemCorrelationMatrix.solveLinearSystem(r);

	double dotRTRinvR = r.dot(R_inv_r);
	double dotRTRinvI = r.dot(R_inv_I);
	double dotITRinvI = vectorOfOnes.dot(R_inv_I);

	double term1 = pow( (dotRTRinvI - 1.0 ),2.0)/dotITRinvI;

	if(fabs(dotRTRinvI - 1.0) < 10E-10) {

		term1 = 0.0;
	}

	double term2 = 1.0 - dotRTRinvR;

	if(fabs(dotRTRinvR - 1.0) < 10E-10) {

		term2 = 0.0;
	}

	double term3 = term2 + term1;

	*sSqrOutput = sigmaSquared* term3;

}

mat KrigingModel::getCorrelationMatrix(void) const{

	return linearSystemCorrelationMatrix.getMatrix();
}

void KrigingModel::checkIfDataIsNormalized() {
	if (!ifNormalized) {
		ModelLogger::getInstance().log(ERROR, "Data has not been normalized.");
		throw std::runtime_error("Data has not been normalized.");
	}
}

double KrigingModel::calculateValidationErrorForGivenHyperparameters(vec hyperParameters) {

	checkIfDataIsRead();
	checkIfDataIsNormalized();

	if (numberOfSamples == 0) {
		throw std::invalid_argument("Number of samples must be greater than zero.");
	}

	if( data.getNumberOfSamplesValidation() == 0){

		throw std::invalid_argument("There are no validation samples.");
	}

	// Set the hyperparameters for the correlation function
	correlationFunction.setHyperParameters(hyperParameters);

	// Update auxiliary fields
	updateAuxilliaryFields();

	// Check if the factorization of the correlation matrix was successful
	if (!linearSystemCorrelationMatrix.isFactorizationDone()) {
		return std::numeric_limits<double>::lowest(); // Return a large negative value if factorization fails
	}

	return calculateValidationError();
}



double KrigingModel::calculateLikelihoodFunction(vec hyperParameters) {

	checkIfDataIsRead();
	checkIfDataIsNormalized();

	if (numberOfSamples == 0) {
		throw std::invalid_argument("Number of samples must be greater than zero.");
	}

	// Set the hyperparameters for the correlation function
	correlationFunction.setHyperParameters(hyperParameters);

	// Update auxiliary fields
	updateAuxilliaryFields();

	// Check if the factorization of the correlation matrix was successful
	if (!linearSystemCorrelationMatrix.isFactorizationDone()) {
		return std::numeric_limits<double>::lowest(); // Return a large negative value if factorization fails
	}

	// Calculate the log determinant of the correlation matrix
	double logdetR = linearSystemCorrelationMatrix.calculateLogDeterminant();
	double NoverTwo = static_cast<double>(numberOfSamples) / 2.0;

	double likelihoodValue = 0.0;

	// Calculate the likelihood value if sigmaSquared is positive
	if (sigmaSquared > 0) {
		double logSigmaSqr = log(sigmaSquared);
		likelihoodValue = (-NoverTwo) * logSigmaSqr;
		likelihoodValue -= 0.5 * logdetR;
	} else {
		likelihoodValue = std::numeric_limits<double>::lowest(); // Return a large negative value if sigmaSquared is not positive
	}

	return likelihoodValue;
}

void KrigingModel::train() {
	if (!ifInitialized) {
		ModelLogger::getInstance().log(ERROR, "KrigingModel::train: Kriging model is not initialized. Cannot proceed with training.");
		throw std::runtime_error("KrigingModel::train: Kriging model is not initialized. Cannot proceed with training.");
	}

	ModelLogger::getInstance().log(INFO, "Model training for the Kriging model started...");

	// Check if warm start file should be read
	if (ifReadWarmStartFile) {
		loadHyperParameters();
	} else {
		// Validate dimension
		if (dimension == 0) {
			ModelLogger::getInstance().log(ERROR, "KrigingModel::train: Invalid dimension detected. The dimension must be set.");
			throw std::runtime_error("KrigingModel::train: Invalid dimension detected. The dimension must be set.");
		}

		if(ifHasValidationSamples){
			setObjectiveFunctionTypeForModelTraining(VALIDATION_ERROR);
		}


		// Setting up bounds for hyperparameter optimization
		vec lb(2 * dimension);
		vec ub(2 * dimension);
		for (unsigned int i = 0; i < dimension; i++) ub(i) = 20.0;
		for (unsigned int i = dimension; i < 2 * dimension; i++) {
			ub(i) = 2.0;
			lb(i) = 1.0;
		}

		double globalBestError = std::numeric_limits<double>::max();

		unsigned int problemDimForInternalOptimization = 2*dimension;
		unsigned int numberOfNewIndividualInGeneration = 200*dimension;
		unsigned int numberOfDeathsInGeneration        = 180*dimension;
		unsigned int initialPopulationSize             = std::min(2 * dimension * 100, 5000u);

		unsigned int numberOfGenerations = numberOfTrainingIterations / numberOfNewIndividualInGeneration;
		if (numberOfGenerations == 0) {
			numberOfGenerations = 1;
		}

		ModelLogger::getInstance().log(INFO, "Settings for Kriging internal optimization:");
		ModelLogger::getInstance().log(INFO, "Maximum number of iterations = " + std::to_string(numberOfTrainingIterations));
		ModelLogger::getInstance().log(INFO, "Initial population size = " + std::to_string(initialPopulationSize));
		ModelLogger::getInstance().log(INFO, "Number of generations = " + std::to_string(numberOfGenerations));
		ModelLogger::getInstance().log(INFO, "Number of threads = " + std::to_string(numberOfThreads));

		numberOfTrainingIterations = numberOfTrainingIterations / numberOfThreads;
		ModelLogger::getInstance().log(INFO, "Number of iterations per thread: " + std::to_string(numberOfTrainingIterations));
		vec bestSolution;
#ifdef OPENMP_SUPPORT
		omp_set_num_threads(numberOfThreads);
#pragma omp parallel for
#endif
		for (unsigned int thread = 0; thread < numberOfThreads; thread++) {
			KrigingHyperParameterOptimizer parameterOptimizer;
			parameterOptimizer.objectiveFunctionType = objectiveFunctionType;
			parameterOptimizer.setDimension(problemDimForInternalOptimization);
			parameterOptimizer.initializeKrigingModelObject(*this);
			parameterOptimizer.setBounds(lb, ub);
			parameterOptimizer.setNumberOfNewIndividualsInAGeneration(numberOfNewIndividualInGeneration);
			parameterOptimizer.setNumberOfDeathsInAGeneration(numberOfDeathsInGeneration);

			parameterOptimizer.setInitialPopulationSize(initialPopulationSize);
			parameterOptimizer.setMutationProbability(0.1);
			parameterOptimizer.setMaximumNumberOfGeneratedIndividuals(numberOfTrainingIterations);


			parameterOptimizer.setNumberOfGenerations(numberOfGenerations);

			parameterOptimizer.optimize();

			EAIndividual threadBestSolution = parameterOptimizer.getSolution();
			vec optimizedHyperParameters = threadBestSolution.getGenes();

#ifdef OPENMP_SUPPORT
#pragma omp critical
#endif
			{
				double bestSolutionLikelihood = threadBestSolution.getObjectiveFunctionValue();
				ModelLogger::getInstance().log(INFO, "Best likelihood at local thread = " + std::to_string(bestSolutionLikelihood));
				if (bestSolutionLikelihood < globalBestError) {
					bestSolution = optimizedHyperParameters;
					globalBestError = bestSolutionLikelihood;
				}
			}
		} /* end of for loop for training at thread level*/

#ifdef OPENMP_SUPPORT
		omp_set_num_threads(1);
#endif

		// Set the best hyperparameters found
		ModelLogger::getInstance().log(INFO, "Solution after model training");
		ModelLogger::getInstance().log(INFO, "Theta = " + bestSolution.head(dimension).toString());
		ModelLogger::getInstance().log(INFO, "Gamma = " + bestSolution.tail(dimension).toString());

		correlationFunction.setHyperParameters(bestSolution);

		// Save hyperparameters if warm start is enabled
		if (ifWriteWarmStartFile) {
			saveHyperParameters();
		}
	}

	if(objectiveFunctionType == VALIDATION_ERROR){
		ModelLogger::getInstance().log(INFO, "Kriging training has been performed using validation error");
		data.revertBackToFullData();
		mat X = data.getInputMatrix();
		numberOfSamples = data.getNumberOfSamples();
		correlationFunction.setInputSampleMatrix(X);

		if(ifUsesLinearRegression){

			vec ys = data.getOutputVector();
			vec ysLinearRegression = linearModel.interpolateAll(X);
			ys = ys - ysLinearRegression;
			data.setOutputVector(ys);
		}

		resizeDataObjects();

	}
	else{
		ModelLogger::getInstance().log(INFO, "Kriging training has been performed using MLE");

	}

	// Update auxiliary fields after training
	updateAuxilliaryFields();

	ModelLogger::getInstance().log(INFO, "Model training is done.");

	ifModelTrainingIsDone = true;
}


void KrigingHyperParameterOptimizer::initializeKrigingModelObject(KrigingModel input){

	if (!input.ifDataIsRead) {
		throw std::runtime_error("KrigingHyperParameterOptimizer::initializeKrigingModelObject: Data has not been read.");
	}
	if (!input.ifNormalized) {
		throw std::runtime_error("KrigingModel::updateAuxilliaryFields: Data has not been normalized.");
	}
	if (!input.ifInitialized) {
		throw std::runtime_error("KrigingModel::updateAuxilliaryFields: Model not been initialized.");
	}


	KrigingModelForCalculations = input;

	ifModelObjectIsSet = true;

}

double KrigingHyperParameterOptimizer::calculateObjectiveFunctionInternal(const vec& input){


	if(objectiveFunctionType == VALIDATION_ERROR){
		return KrigingModelForCalculations.calculateValidationErrorForGivenHyperparameters(input);
	}
	else if(objectiveFunctionType == MAXIMUM_LIKELIHOOD){
		return -1.0* KrigingModelForCalculations.calculateLikelihoodFunction(input);
	}
	else{
		throw std::runtime_error("Not valid objective function type for Kriging parameter training");
	}
}


void KrigingHyperParameterOptimizer::printInternalObject() const{

	KrigingModelForCalculations.printSurrogateModel();

}

} /* Namespace Rodop */
