#include "./INCLUDE/rodeo.h"
#include "../Optimizers/INCLUDE/optimization.hpp"
#include "../LinearAlgebra/INCLUDE/matrix.hpp"
#include "../ObjectiveFunctions/INCLUDE/objective_function.hpp"
#include "../ObjectiveFunctions/INCLUDE/constraint_functions.hpp"
#include "../Bounds/INCLUDE/bounds.hpp"
#include <filesystem>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#define CHANGE_DIR _chdir
#else
#include <unistd.h>
#define CHANGE_DIR chdir
#endif

bool changeDirectory(const std::string& directory) {
	if (CHANGE_DIR(directory.c_str()) != 0) {
		std::cerr << "Error: Could not change directory to " << directory << std::endl;
		return false;
	}
	return true;
}


namespace Rodop{

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


void RobustDesignOptimizer::setName(const std::string& nameInput) {
	// Check if the input name is empty and throw an exception if it is
	if (nameInput.empty()) {
		throw std::invalid_argument("Name cannot be empty.");
	}

	// Set the name for the optimizer and the internal name
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
	boxConstraints.setBounds(lowerBounds,upperBounds);

	optimizer.setBoxConstraints(boxConstraints);
	areBoxConstraintsSpecified = true;


}

void RobustDesignOptimizer::setNameOfTrainingDataFile(std::string filename){

	if(filename.empty()){
		throw std::invalid_argument("Empty file name for the training data");
	}
	objectiveFunctionTrainingFilename = name;
}

void RobustDesignOptimizer::setDimension(unsigned int dim){

	dimension = dim;
	optimizer.setDimension(dimension);
	isDimensionSpecified = true;
}

void RobustDesignOptimizer::setObjectiveFunction(ObjectiveFunctionPtr function, std::string functionName, std::string filename){

	if(filename.empty()){

		throw std::invalid_argument("Empty file name for the training data");
	}
	if(functionName.empty()){

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
	objectiveFunctionDefinition.name = functionName;


	objectiveFunctionToAdd.setParametersByDefinition(objectiveFunctionDefinition);

	optimizer.addObjectFunction(objectiveFunctionToAdd);

	isObjectiveFunctionSpecified = true;

}

// Helper function to trim whitespace from a string
std::string trim(const std::string& str) {
	size_t start = str.find_first_not_of(" \t\n\r");
	size_t end = str.find_last_not_of(" \t\n\r");
	return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end + 1);
}

ParsedConstraintExpression parseExpression(const std::string& input) {
	// Trim the input to remove leading/trailing whitespace
	std::string trimmedInput = trim(input);
	std::istringstream stream(trimmedInput);
	ParsedConstraintExpression result;

	// Read the name part
	if (!(stream >> result.name)) {
		throw std::invalid_argument("Failed to parse the name part from expression: '" + input + "'");
	}

	// Read the inequality part
	if (!(stream >> result.inequality)) {
		throw std::invalid_argument("Failed to parse the inequality part from expression: '" + input + "'");
	}

	// Validate the inequality part
	if (result.inequality != "<" && result.inequality != ">") {
		throw std::invalid_argument("Invalid inequality operator in expression: '" + input + "'");
	}

	// Read the floating point number part
	if (!(stream >> result.value)) {
		throw std::invalid_argument("Failed to parse the floating point number from expression: '" + input + "'");
	}

	return result;
}


void RobustDesignOptimizer::addConstraint(ObjectiveFunctionPtr function, std::string expression, std::string filename) {
	// Check for a valid filename
	if (filename.empty()) {
		throw std::invalid_argument("Empty file name for the training data.");
	}

	// Check if the function name (constraint) is empty
	if (expression.empty()) {
		throw std::invalid_argument("Expression for the constraint is empty.");
	}

	// Check if a valid function pointer is passed
	if (function == nullptr) {
		throw std::invalid_argument("Null function pointer provided.");
	}

	// Parse the constraint expression
	ParsedConstraintExpression exp = parseExpression(expression);

	// Ensure the parsed expression has a valid constraint name
	if (exp.name.empty()) {
		throw std::invalid_argument("Constraint name parsed from expression is empty.");
	}

	// Ensure the inequality is valid
	if (exp.inequality != "<" && exp.inequality != ">") {
		throw std::invalid_argument("Invalid inequality in constraint expression.");
	}

	// Add the function pointer and filename to the appropriate vectors
	constraintFunctionPtr.push_back(function);
	constraintsTrainingDataFilename.push_back(filename);

	// Create a new constraint function object
	ConstraintFunction constraint;
	constraint.setFunctionPtr(function);

	// Set up the objective function definition
	ObjectiveFunctionDefinition definition;
	definition.nameHighFidelityTrainingData = filename;
	definition.name = exp.name;

	// Assign the definition to the constraint
	constraint.setParametersByDefinition(definition);

	// Define the constraint's inequality and value
	ConstraintDefinition constraintExpression;
	constraintExpression.constraintName = exp.name;
	constraintExpression.inequalityType = exp.inequality;
	constraintExpression.value = exp.value;
	constraintExpression.ID = numberOfConstraints;
	numberOfConstraints++;

	// Set the constraint's definition and dimension
	constraint.setConstraintDefinition(constraintExpression);
	constraint.setDimension(dimension);

	// Add the constraint to the optimizer
	optimizer.addConstraint(constraint);

	// Optionally, add logging here to record the success of adding the constraint
	// DriverLogger::getInstance().log(INFO, "Constraint '" + exp.name + "' added successfully.");
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

    // Validate if constraint function pointers are properly set
    if (constraintFunctionPtr.empty()) {
        throw std::runtime_error("Constraint function pointers are not set.");
    }

    for (unsigned int iConstraint = 0; iConstraint < numberOfConstraints; iConstraint++) {

        mat trainingDataForConstraint;

        // Loop through each sample and evaluate the constraints
        for (unsigned int i = 0; i < numberOfSamplesForDoE; i++) {
            vec x = samplesInput.getRow(i);

            try {
                // Evaluate the constraint function
                double f = constraintFunctionPtr[iConstraint](x.getPointer());

                // Add the constraint result to the sample data
                vec sample = x;
                sample.push_back(f);

                // Add the sample to the training data for this constraint
                trainingDataForConstraint.addRow(sample);

            } catch (const std::exception& e) {
                std::cerr << "Error evaluating constraint " << iConstraint
                          << " for sample " << i << ": " << e.what() << std::endl;
                throw;
            }
        }

        try {
            // Save all training data for this constraint at once, after collecting all rows
            trainingDataForConstraint.saveAsCSV(constraintsTrainingDataFilename[iConstraint]);
        } catch (const std::exception& e) {
            std::cerr << "Error saving training data for constraint " << iConstraint
                      << " to file: " << e.what() << std::endl;
            throw;
        }
    }
}


void RobustDesignOptimizer::generateDoESamplesInput() {
	// Check if DoE type is valid and perform appropriate sampling
	if (DoEType == "random") {
		samplesInput.resize(numberOfSamplesForDoE, dimension);
		samplesInput.fillRandom(lowerBounds, upperBounds);
	} else if (DoEType == "lhs") {
		samplesInput.resize(numberOfSamplesForDoE, dimension);
		samplesInput.fillRandomLHS(lowerBounds, upperBounds);
	} else {
		throw std::invalid_argument("Invalid DoE type: " + DoEType);
	}
}

void RobustDesignOptimizer::performDoE(void) {

	generateDoESamplesInput();

	// Check if the objective function pointer is set
	if (objectiveFunctionPtr == nullptr) {
		throw std::runtime_error("Objective function pointer is null.");
	}

	if(numberOfSamplesForDoE == 0){
		throw std::runtime_error("Number of samples must be specified for the DoE.");
	}

	vec functionValues(numberOfSamplesForDoE);
	mat trainingData;
	for (unsigned int i = 0; i < numberOfSamplesForDoE; ++i) {
		vec x = samplesInput.getRow(i);
		try {
			// Try to evaluate the objective function
			functionValues(i) = objectiveFunctionPtr(x.getPointer());
		} catch (const std::exception& e) {
			// Handle the exception, possibly logging the error and rethrowing it
			std::cerr << "Error evaluating the objective function at sample " << i << ": " << e.what() << std::endl;
			throw std::runtime_error("DoE process will be killed.");
		}

		// Add the evaluated function value to the sample
		vec sampleToAdd = x;
		sampleToAdd.push_back(functionValues(i));
		trainingData.addRow(sampleToAdd);

		try {
			// Save the training data
			trainingData.saveAsCSV(objectiveFunctionTrainingFilename);
		} catch (const std::exception &e) {
			throw std::runtime_error("DoE process will be killed.");
		}
	}

	// If constraints exist, perform DoE for them
	if (numberOfConstraints > 0) {
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
	if (!isObjectiveFunctionSpecified) {
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


} /* Namespace Rodop */
