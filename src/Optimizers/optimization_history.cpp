#include <cassert>
#include <algorithm>
#include <set>
#include "./INCLUDE/optimization_history.hpp"
#include "./INCLUDE/optimization_logger.hpp"
#include "../Design/INCLUDE/design.hpp"
#include "../LinearAlgebra/INCLUDE/matrix.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"


namespace Rodop{


void OptimizationHistory::addConstraintName(const std::string& name) {
	// Ensure the constraint name is not empty
	if (name.empty()) {
		throw std::invalid_argument("Constraint name cannot be empty.");
	}

	// Optional: Check for duplicate constraint names
	if (std::find(constraintNames.begin(), constraintNames.end(), name) != constraintNames.end()) {
		throw std::invalid_argument("Constraint name already exists: " + name);
	}

	// Add the constraint name to the list
	constraintNames.push_back(name);
}


void OptimizationHistory::reset(void){
	constraintNames.clear();
	data.reset();
	dimension = 0;
	filename.clear();
}

void OptimizationHistory::setDimension(unsigned int dim){
	dimension = dim;
}

void OptimizationHistory::setData(mat dataIn){

	if(dimension == 0){
		throw std::runtime_error("OptimizationHistory::setData: dimension cannot be zero.");
	}

	if (dataIn.isEmpty()) {
		throw std::invalid_argument("OptimizationHistory::setData: Empty data.");
	}

	unsigned int numberOfEntries = dimension + 1 + static_cast<unsigned int>(constraintNames.size()) + 2;

	if (dataIn.getNCols() != numberOfEntries) {
		std::cout<<"number of entries = " << numberOfEntries <<"\n";
		std::cout<<"number of columns of the data matrix = " << dataIn.getNCols() <<"\n";
		throw std::runtime_error("OptimizationHistory::setData: number of columns do not match.");
	}

	data = dataIn;

}

void OptimizationHistory::setParameterNames(vector<string> names) {

	if(dimension == 0){
		throw std::runtime_error("OptimizationHistory::setParameterNames: Dimension must be set first.");
	}
	if (names.empty()) {
		throw std::invalid_argument("The parameter names vector is empty.");
	}

	if(names.size() != dimension){
		std::cout<<"Name vector size = " << names.size() << "\n";
		std::cout<<"Dimension = " << dimension << "\n";
		throw std::invalid_argument("The parameter names vector size does not match with dimension.");
	}

	std::set<std::string> nameSet;
	for (const auto& name : names) {
		if (!nameSet.insert(name).second) {
			throw std::invalid_argument("The parameter names vector contains duplicate names.");
		}
	}
	parameterNames = names;

	OptimizationLogger::getInstance().log(INFO,"OptimizationHistory: setParameterNames");
	OptimizationLogger::getInstance().log(INFO,"OptimizationHistory: number of parameters = " + std::to_string(names.size()));
	for (const auto& name : names) {
		OptimizationLogger::getInstance().log(INFO,"OptimizationHistory: set parameter name = " + name);

	}


}


mat OptimizationHistory::getData(void) const{
	return data;
}

double OptimizationHistory::getCrowdingFactor(void) const{

	return crowdingFactor;
}

vec OptimizationHistory::getObjectiveFunctionValues(void) const{
	assert(dimension>0);
	return data.getCol(dimension);
}

vec OptimizationHistory::getFeasibilityValues(void) const{
	assert(data.getNCols() >0);
	return data.getCol(data.getNCols()-1);

}

void OptimizationHistory::setObjectiveFunctionName(string name){

	assert(!name.empty());
	objectiveFunctionName = name;

}




vector<string> OptimizationHistory::setHeader() const {
	vector<string> fileHeader;

	// Ensure parameter names size matches dimension if names are provided
	if (!parameterNames.empty() && parameterNames.size() != dimension) {
		std::cout<<"Dimension = " << dimension << "\n";
		for (const auto& name : parameterNames) {
			std::cout<< name << "\n";
		}

		throw std::invalid_argument("The parameter names vector size is different than problem dimension.");
	}

	// Reserve space for file header to avoid multiple allocations
	fileHeader.reserve(dimension + constraintNames.size() + 3);  // parameters + objective + constraints + improvement + feasibility

	// Set parameter names or default variable names
	if (!parameterNames.empty()) {
		fileHeader.insert(fileHeader.end(), parameterNames.begin(), parameterNames.end());
	} else {
		for (unsigned int i = 0; i < dimension; ++i) {
			fileHeader.push_back("x" + std::to_string(i + 1));
		}
	}

	// Set objective field
	fileHeader.push_back("Objective");

	// Set constraint names
	fileHeader.insert(fileHeader.end(), constraintNames.begin(), constraintNames.end());

	// Set remaining fields
	fileHeader.push_back("Improvement");
	fileHeader.push_back("Feasibility");

#if 0
	for (const auto& name : fileHeader) {
		OptimizationLogger::getInstance().log(INFO, "OptimizationHistory: file header entity = " + name);
	}
#endif
	return fileHeader;
}


void OptimizationHistory::saveOptimizationHistoryFile() {

	// Check if the data is empty, and throw an exception if it is
	if (data.isEmpty()) {
		throw std::runtime_error("Cannot save optimization history: Data is empty.");
	}

	// Get the file header
	vector<std::string> fileHeader = setHeader();

	// Save the data to a CSV file with precision 6 and the generated header
	data.saveAsCSV(filename, 6, fileHeader);

	// Log the save operation
	OptimizationLogger::getInstance().log(INFO, "Optimization history saved to file: " + filename);
}


void OptimizationHistory::updateOptimizationHistory(Design d) {

	// Ensure the design parameters match the problem's dimension
	if (d.designParameters.getSize() != dimension) {
		throw std::invalid_argument("Design parameter size does not match problem dimension.");
	}

	// Calculate the number of entries for the new row
	unsigned int numberOfEntries = dimension + 1 + static_cast<unsigned int>(constraintNames.size()) + 2;

	// Initialize the new row with the correct number of entries
	vec newRow(numberOfEntries);

	// Fill the row with design parameters
	for (unsigned int i = 0; i < dimension; ++i) {
		newRow(i) = d.designParameters(i);
	}

	// Fill the objective value
	newRow(dimension) = d.trueValue;

	// Fill the constraint values
	for (unsigned int i = 0; i < static_cast<unsigned int>(constraintNames.size()); ++i) {
		newRow(i + dimension + 1) = d.constraintTrueValues(i);
	}

	// Fill the improvement value
	newRow(dimension + static_cast<unsigned int>(constraintNames.size()) + 1) = d.improvementValue;

	// Fill the feasibility status (1.0 for feasible, 0.0 for infeasible)
	newRow(dimension + static_cast<unsigned int>(constraintNames.size()) + 2) = d.isDesignFeasible ? 1.0 : 0.0;

	// Add the new row to the data
	data.addRow(newRow, -1);

	// Save the updated optimization history to a file
	saveOptimizationHistoryFile();

	// Log the update
	OptimizationLogger::getInstance().log(INFO, "Optimization history updated with new design.");
}



double OptimizationHistory::calculateInitialImprovementValue() const {
	unsigned int N = data.getNRows();

	if (N == 0) {
		throw std::runtime_error("No data available to calculate initial improvement value.");
	}

	vec objectiveFunctionValues = getObjectiveFunctionValues();
	vec feasibilityValues = getFeasibilityValues();

	bool ifFeasibleDesignFound = false;
	double bestFeasibleObjectiveFunctionValue = std::numeric_limits<double>::max();  // Using max value as initial

	// Loop through all rows to find the best feasible objective function value
	for (unsigned int i = 0; i < N; ++i) {
		if (feasibilityValues(i) > 0.0 && objectiveFunctionValues(i) < bestFeasibleObjectiveFunctionValue) {
			ifFeasibleDesignFound = true;
			bestFeasibleObjectiveFunctionValue = objectiveFunctionValues(i);
		}
	}

	// Return the best feasible value if found, otherwise a large negative value
	if (ifFeasibleDesignFound) {
		return bestFeasibleObjectiveFunctionValue;
	} else {
		return -std::numeric_limits<double>::max();  // Returning a large negative value if no feasible design found
	}
}

void OptimizationHistory::print() const {

	std::cout << "Dimension: " << dimension << "\n";
	std::cout << "Objective function name: " << objectiveFunctionName << "\n";

	if (!constraintNames.empty()) {
		std::cout << "Constraint names: \n";
		for (const auto& name : constraintNames) {
			std::cout << name << "\n";
		}
	}

	// Print data object with header or description
	data.print("Data = \n");
}


void OptimizationHistory::calculateCrowdingFactor() {
	// Ensure there is valid data to process
	if (data.getNRows() <= 0 || dimension <= 0 || numberOfDoESamples <= 0) {
		throw std::runtime_error("Invalid data, dimension, or DoE sample size.");
	}

	// Extract the subset of data to process based on the number of samples
	mat dataToProcess;
	if (data.getNRows() >= numberOfDoESamples) {
		dataToProcess = data.submat(data.getNRows() - numberOfDoESamples, data.getNRows() - 1, 0, dimension - 1);
	} else {
		dataToProcess = data.submat(0, 0, data.getNRows() - 1, dimension - 1);
	}

	double sum = 0.0;
	unsigned int numRows = dataToProcess.getNRows();

	// Calculate the pairwise L1 norm differences between rows
	for (unsigned int i = 0; i < numRows; ++i) {
		vec x1 = dataToProcess.getRow(i);

		for (unsigned int j = i + 1; j < numRows; ++j) {  // Skip i == j (self-pairs)
			vec x2 = dataToProcess.getRow(j);
			vec d = x1 - x2;
			double normDiff = d.norm(L1);
			sum += normDiff * 2;  // Since d(x1, x2) == d(x2, x1), we can double the contribution
		}
	}

	// Number of unique pairs is numRows * (numRows - 1) / 2
	unsigned int numPairs = numRows * (numRows - 1) / 2;

	// Calculate the average crowding factor
	if (numPairs > 0) {
		crowdingFactor = sum / (numPairs * 2);  // Dividing by 2 because we doubled the sum
	} else {
		crowdingFactor = 0.0;  // In case there are no valid pairs
	}
}


} /* Namespace Rodop */
