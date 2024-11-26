
#include <cassert>
#include <fstream>

#include "./INCLUDE/globalOptimalDesign.hpp"
#include "./INCLUDE/optimization_logger.hpp"
#include "../LinearAlgebra/INCLUDE/matrix.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"

namespace Rodop{

void GlobalOptimalDesign::setBoxConstraints(const Bounds& input) {
    // Ensure that the input bounds are valid and match the problem dimension
    if (!input.areBoundsSet()) {
        throw std::invalid_argument("Box constraints are not properly set.");
    }

    if (dimension != input.getDimension()) {
        throw std::invalid_argument("Mismatch between box constraint dimension and design dimension.");
    }

    // Set the box constraints
    boxConstraints = input;

    // Optional: Log the successful setting of box constraints
    OptimizationLogger::getInstance().log(INFO, "Box constraints successfully set.");
}

void GlobalOptimalDesign::validateInputs(const mat& historyFile) const {
	if (historyFile.getNRows() == 0) {
		throw std::invalid_argument("The history file is empty.");
	}

	if (!boxConstraints.areBoundsSet()) {
		throw std::logic_error("Box constraints are not set.");
	}

	if (dimension <= 0) {
		throw std::logic_error("Dimension must be greater than 0.");
	}
}


unsigned int GlobalOptimalDesign::findBestDesignIndex(const mat& historyFile, bool& isFeasibleDesignFound) const {
	unsigned int numSamples = historyFile.getNRows();
	unsigned int lastColIndex = historyFile.getNCols() - 1;

	double bestObjectiveValue = std::numeric_limits<double>::max();
	unsigned int bestDesignIndex = 0;
	isFeasibleDesignFound = false;

	// Iterate through each sample in the history file
	for (unsigned int i = 0; i < numSamples; i++) {
		double feasibility = historyFile(i, lastColIndex);
		double objectiveValue = historyFile(i, dimension);

		// Check if the design is feasible and better than the current best
		if (feasibility > 0.0 && objectiveValue < bestObjectiveValue) {
			isFeasibleDesignFound = true;
			bestObjectiveValue = objectiveValue;
			bestDesignIndex = i;
		}
	}

	// If no feasible design is found, choose the best objective function value
	if (!isFeasibleDesignFound) {
		vec objectiveValues = historyFile.getCol(dimension);

		bestDesignIndex = objectiveValues.findMinIndex();
	}

	return bestDesignIndex;
}


void GlobalOptimalDesign::extractDesignData(const vec& bestSample) {
	vec dv = bestSample.head(dimension);

	tag = "Global optimum design";
	designParameters = dv;
	trueValue = bestSample(dimension);
	improvementValue = bestSample(bestSample.getSize() - 2);

	vec constraintValues(numberOfConstraints);
	for (unsigned int i = 0; i < numberOfConstraints; i++) {
		constraintValues(i) = bestSample(i + dimension + 1);
	}

	constraintTrueValues = constraintValues;

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	designParametersNormalized = dv.normalizeVector(lb,ub);

}

void GlobalOptimalDesign::setGlobalOptimalDesignFromHistoryFile(const mat& historyFile) {
	validateInputs(historyFile);

	bool isFeasibleDesignFound = false;
	unsigned int bestDesignIndex = findBestDesignIndex(historyFile, isFeasibleDesignFound);

	vec bestSample = historyFile.getRow(bestDesignIndex);
	isDesignFeasible = isFeasibleDesignFound;
	ID = bestDesignIndex;

	extractDesignData(bestSample);
}


void GlobalOptimalDesign::setGradientGlobalOptimumFromTrainingData(const std::string &nameOfTrainingData) {
    // Ensure that the provided training data name is not empty
    if (nameOfTrainingData.empty()) {
        throw std::invalid_argument("The name of the training data cannot be empty.");
    }

    // Ensure that the dimension and design parameters are valid
    if (dimension <= 0) {
        throw std::logic_error("Dimension must be greater than 0.");
    }

    if (designParameters.getSize() == 0) {
        throw std::logic_error("Design parameters cannot be empty.");
    }

    mat trainingDataToSearch;
    trainingDataToSearch.readFromCSV(nameOfTrainingData);


    // Check if the training data has enough rows and columns
    if (trainingDataToSearch.getNRows() == 0 || trainingDataToSearch.getNCols() < dimension) {
        throw std::runtime_error("Training data is empty or does not match the expected dimension.");
    }

    // Extract the input part of the training data (only the design parameter columns)
    mat trainingDataInput = trainingDataToSearch.submat(0, trainingDataToSearch.getNRows() - 1, 0, dimension - 1);

 //   trainingDataInput.print("trainingDataInput");

    // Find the index of the global optimal design in the training data
    constexpr double tolerance = 1e-5; // Define tolerance as a constant
    int indexOfTheGlobalOptimalDesignInTrainingData = trainingDataInput.findRowIndex(designParameters, tolerance);

    // Check if the row was found
    if (indexOfTheGlobalOptimalDesignInTrainingData == -1) {
        std::ostringstream oss;
        oss << "Could not find the design parameters in the training data.\n";
        oss << "design Parameters = " << designParameters.toString();
        trainingDataInput.print("training data input");
        throw std::runtime_error(oss.str());
    }

    // Extract the gradient vector from the corresponding row
    vec bestRow = trainingDataToSearch.getRow(indexOfTheGlobalOptimalDesignInTrainingData);
    if (bestRow.getSize() < dimension * 2) {
        throw std::runtime_error("Training data row does not have enough data to extract the gradient.");
    }

    vec gradientVector = bestRow.tail(dimension);  // Extract the gradient from the tail of the row

    // Set the gradient vector
    gradient = gradientVector;
}


void GlobalOptimalDesign::saveToXMLFile(void) const {
    if (xmlFileName.empty()) {
        throw std::runtime_error("XML file name is empty. Cannot save XML.");
    }

    std::ofstream file(xmlFileName);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + xmlFileName);
    }

    std::string text = generateXmlString();
    file << text;
    file.close();
}


string GlobalOptimalDesign::generateXml(const std::string& elementName, const int& value) const{
	std::ostringstream xml;
	xml << std::fixed << "<" << elementName << ">" << value << "</" << elementName << ">";
	return xml.str();
}

string GlobalOptimalDesign::generateXml(const std::string& elementName, const double& value) const{
	std::ostringstream xml;
	xml << std::fixed << "<" << elementName << ">" << value << "</" << elementName << ">";
	return xml.str();
}
string GlobalOptimalDesign::generateXml(const std::string& elementName, const string& value) const{
	std::ostringstream xml;
	xml << std::fixed << "<" << elementName << ">" << value << "</" << elementName << ">";
	return xml.str();
}

std::string GlobalOptimalDesign::generateXmlVector(const std::string& name, const vec& data) const {

	assert(data.getSize() > 0);
	std::ostringstream xml;

	xml << "<" << name << ">\n";
	for (unsigned int i=0; i<data.getSize(); i++) {
		xml << std::fixed << "\t<item>" << data(i) << "</item>\n";
	}

	xml << "</" << name << ">";

	return xml.str();
}


string GlobalOptimalDesign::generateXmlString() const {
	// Initialize the XML string with the root element
	string result = "<GlobalOptimalDesign>\n";

	// Add design ID
	result += generateXml("DesignID", ID) + "\n";

	// Add objective function value
	result += generateXml("ObjectiveFunction", trueValue) + "\n";

	// Add design parameters
	result += generateXmlVector("DesignParameters", designParameters) + "\n";

	// Add constraint values if they exist
	if (!constraintTrueValues.isEmpty()) {
		result += generateXmlVector("ConstraintValues", constraintTrueValues) + "\n";
	}

	// Add feasibility status
	if(isDesignFeasible){
		string yes = "YES";
		result += generateXml("Feasibility", yes) + "\n";
	}
	else{
		string no = "NO";
		result += generateXml("Feasibility", no) + "\n";
	}

	// Close the root element
	result += "</GlobalOptimalDesign>\n";

	return result;
}


bool GlobalOptimalDesign::checkIfGlobalOptimaHasGradientVector(void) const{

	if(gradient.isEmpty()) {
		return false;
	}
	else{
		if(gradient.is_zero()){
			return false;
		}
		else{
			return true;
		}
	}
}

} /* Namespace Rodop */
