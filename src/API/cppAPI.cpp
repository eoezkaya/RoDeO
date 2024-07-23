#include "./INCLUDE/rodeo.h"
#include "../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../Auxiliary/INCLUDE/print.hpp"
#include "../Bounds/INCLUDE/bounds.hpp"
#include <cassert>
#include <filesystem>


RobustDesignOptimizer::RobustDesignOptimizer(){
	optimizer.setAPIUseOn();
	name = "OptimizationStudy";
	optimizer.setName(name);
}

void RobustDesignOptimizer::print(void){

	std::cout<<"Name      = "<<name<<"\n";
	std::cout<<"Dimension = "<<dimension<<"\n";
	std::cout<<"Training data file name = "<<objectiveFunctionTrainingFilename<<"\n";

}


void RobustDesignOptimizer::setName(const string nameInput){

	assert(!nameInput.empty());
	name = nameInput;
	optimizer.setName(name);

}


void RobustDesignOptimizer::setCurrentWorkingDirectory(string directory){
	if (directory.empty() || !std::filesystem::is_directory(directory)) {
		throw std::invalid_argument("Current working directory is not valid");
	}

	cwd = directory;

}


void RobustDesignOptimizer::setDoEStrategy(const std::string& input) {
	if (input.empty()) {
		throw std::invalid_argument("DoE strategy cannot be empty");
	}

	std::string lowerInput = input;
	std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);

	if (lowerInput != "lhs" && lowerInput != "random") {
		throw std::invalid_argument("Invalid DoE strategy. Allowed values are 'LHS' or 'Random'");
	}

	DoEType = lowerInput;
}


void RobustDesignOptimizer::setBoxConstraints(double *lb, double *ub){
	if(dimension == 0){
		throw std::invalid_argument("Problem dimension is not specified");
	}

	for(unsigned int i=0;i<dimension; i++){
		if(lb[i] >= ub[i]){
			throw std::invalid_argument("Box constraints are not set properly");

		}

		lowerBounds.push_back(lb[i]);
		upperBounds.push_back(ub[i]);
	}

	Bounds boxConstraints;
	boxConstraints.setDimension(dimension);
	boxConstraints.setBounds(lb,ub);

	optimizer.setBoxConstraints(boxConstraints);
	areBoxConstraintsSpecified = true;


}

void RobustDesignOptimizer::setNameOfTrainingDataFile(std::string name){

	assert(!name.empty());
	objectiveFunctionTrainingFilename = name;
}

void RobustDesignOptimizer::setDimension(unsigned int dim){

	dimension = dim;
	optimizer.setDimension(dimension);
	isDimensionSpecified = true;
}

void RobustDesignOptimizer::setObjectiveFunction(ObjectiveFunctionPtr function, std::string name, std::string filename){

	if(filename.empty()){

		throw std::invalid_argument("Empty file name for the training data");
	}
	if(name.empty()){

		throw std::invalid_argument("Name of the objective function is empty");
	}



	if(function == nullptr){
		throw std::invalid_argument("Null function pointer");
	}


	objectiveFunctionPtr = function;

	ObjectiveFunction objectiveFunctionToAdd;
	objectiveFunctionToAdd.setDimension(dimension);
	objectiveFunctionToAdd.setFunctionPtr(function);
	ObjectiveFunctionDefinition objectiveFunctionDefinition;
	objectiveFunctionTrainingFilename = filename;
	objectiveFunctionDefinition.nameHighFidelityTrainingData = filename;
	objectiveFunctionDefinition.name = name;


	objectiveFunctionToAdd.setParametersByDefinition(objectiveFunctionDefinition);

	optimizer.addObjectFunction(objectiveFunctionToAdd);

	isObjectiveFunctionSpecified = true;

}

ParsedConstraintExpression parseExpression(const std::string& input) {
	std::istringstream stream(input);
	ParsedConstraintExpression result;

	// Read the name part
	if (!(stream >> result.name)) {
		throw std::invalid_argument("Failed to parse the name part.");
	}

	// Read the inequality part
	if (!(stream >> result.inequality)) {
		throw std::invalid_argument("Failed to parse the inequality part.");
	}

	// Validate the inequality part
	if (result.inequality != "<" && result.inequality != ">") {
		throw std::invalid_argument("Invalid inequality operator.");
	}

	// Read the floating point number part
	if (!(stream >> result.value)) {
		throw std::invalid_argument("Failed to parse the floating point number.");
	}

	return result;
}


void RobustDesignOptimizer::addConstraint(ObjectiveFunctionPtr function, std::string expression, std::string filename){
	if(filename.empty()){

		throw std::invalid_argument("Empty file name for the training data");
	}
	if(name.empty()){

		throw std::invalid_argument("Name of the objective function is empty");
	}

	if(function == nullptr){
		throw std::invalid_argument("Null function pointer");
	}

	constraintFunctionPtr.push_back(function);
	constraintsTrainingDataFilename.push_back(filename);

	ConstraintFunction constraint;
	ObjectiveFunctionDefinition definition;
	definition.nameHighFidelityTrainingData = filename;
	ConstraintDefinition constraintExpression;

	constraint.setFunctionPtr(function);

	ParsedConstraintExpression exp = parseExpression(expression);

	constraintExpression.constraintName = exp.name;
	definition.name = constraintExpression.constraintName;

	constraint.setParametersByDefinition(definition);

	constraintExpression.value = exp.value;
	constraintExpression.inequalityType = exp.inequality;
	constraintExpression.ID = numberOfConstraints;
	numberOfConstraints++;
	constraint.setConstraintDefinition(constraintExpression);

	constraint.setDimension(dimension);

	//	constraint.print();

	optimizer.addConstraint(constraint);

}


void RobustDesignOptimizer::setDoEOn(unsigned int nSamples){

	numberOfSamplesForDoE = nSamples;
	startWithDoE = true;
}

void RobustDesignOptimizer::setMaxNumberOfFunctionEvaluations(unsigned int nSamples){

	numberOfFunctionEvaluations = nSamples;
	optimizer.setMaximumNumberOfIterations(numberOfFunctionEvaluations);
}

void RobustDesignOptimizer::performDoEForConstraints(void){

	for(unsigned int iConstraint=0; iConstraint<numberOfConstraints; iConstraint++){

		vec constraintValues(numberOfSamplesForDoE);
		for (unsigned int i = 0; i < numberOfSamplesForDoE; i++) {
			rowvec x = samplesInput.row(i);
			//			x.print();
			double f = constraintFunctionPtr[iConstraint](x.memptr());
			//			std::cout<<"f = " <<f <<"\n";
			constraintValues(i) = f;
		}

		mat trainingDataForConstraint(numberOfSamplesForDoE, dimension + 1);
		for (unsigned int i = 0; i < numberOfSamplesForDoE; i++) {
			for (unsigned int j = 0; j < dimension; j++) {
				trainingDataForConstraint(i, j) = samplesInput(i, j);
			}


			trainingDataForConstraint(i, dimension) = constraintValues(i);
		}

		saveMatToCVSFile(trainingDataForConstraint,constraintsTrainingDataFilename[iConstraint]);

	}

}


void RobustDesignOptimizer::performDoE(void) {
	if(DoEType == "random"){

		samplesInput = generateRandomMatrix(numberOfSamplesForDoE, dimension,
				lowerBounds.data(), upperBounds.data());


	}


	else if(DoEType == "lhs"){

		samplesInput = generateLatinHypercubeMatrix(numberOfSamplesForDoE, dimension, lowerBounds, upperBounds);


	}

	else{

		throw std::invalid_argument("Invalid DoE type.");
	}
	//	sampleInput.print("sampleInput");
	vec functionValues(numberOfSamplesForDoE);
	for (unsigned int i = 0; i < numberOfSamplesForDoE; i++) {
		rowvec x = samplesInput.row(i);
		double f = objectiveFunctionPtr(x.memptr());
		functionValues(i) = f;
	}
	mat trainingData(numberOfSamplesForDoE, dimension + 1);
	for (unsigned int i = 0; i < numberOfSamplesForDoE; i++) {
		for (unsigned int j = 0; j < dimension; j++) {
			trainingData(i, j) = samplesInput(i, j);
		}
		trainingData(i, dimension) = functionValues(i);
	}
	saveMatToCVSFile(trainingData, objectiveFunctionTrainingFilename);
	//	trainingData.print("training data =");

	if(numberOfConstraints > 0){
		performDoEForConstraints();
	}
}

void RobustDesignOptimizer::checkOptimizationSettings() {
	if (numberOfFunctionEvaluations == 0) {
		throw std::invalid_argument(
				"Maximum number of function evaluations is not specified");
	}
	if (!isDimensionSpecified) {
		throw std::invalid_argument("Problem dimension is not specified");
	}
	if (!areBoxConstraintsSpecified) {
		throw std::invalid_argument("Box constraints are not specified");
	}
	if (!this->isObjectiveFunctionSpecified) {
		throw std::invalid_argument("Objective function is not specified");
	}
}

void RobustDesignOptimizer::run(void){

	checkOptimizationSettings();

	if(!changeDirectory(cwd)){

		throw std::invalid_argument("Cannot change to current working directory");
	}

	if(startWithDoE){

		if(numberOfSamplesForDoE == 0){
			throw std::invalid_argument("Number of samples for the DoE is not specified");
		}
		performDoE();
	}


	//	std::cout<<"Running optimization...\n";
	optimizer.performEfficientGlobalOptimization();

}
