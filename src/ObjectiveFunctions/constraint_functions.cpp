#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include <algorithm>  // For std::remove_if


#include "../Bounds/INCLUDE/bounds.hpp"
#include "./INCLUDE/constraint_functions.hpp"
#include "./INCLUDE/objective_function_logger.hpp"


namespace Rodop{

std::string ConstraintDefinition::removeSpacesFromString(std::string inputString) const{

	inputString.erase(std::remove_if(inputString.begin(), inputString.end(), ::isspace), inputString.end());
	return inputString;
}




void ConstraintDefinition::setDefinition(const std::string& definition) {
    // Check if the input definition is empty
    if (definition.empty()) {
        throw std::invalid_argument("Constraint definition is empty.");
    }

    // Find the positions of '>' or '<' in the definition string
    size_t found = definition.find(">");
    size_t found2 = definition.find("<");
    size_t place;
    std::string nameBuf, typeBuf, valueBuf;

    // Determine the inequality operator and position
    if (found != std::string::npos) {
        place = found;
    } else if (found2 != std::string::npos) {
        place = found2;
    } else {
        throw std::invalid_argument("Invalid constraint definition format. Must contain '>' or '<'.");
    }

    // Extract the name, inequality type, and value from the definition string
    nameBuf.assign(definition, 0, place);
    nameBuf = removeSpacesFromString(nameBuf); // Remove spaces from the name

    typeBuf.assign(definition, place, 1); // The inequality symbol (either '>' or '<')

    valueBuf.assign(definition, place + 1, definition.length() - nameBuf.length() - typeBuf.length());
    valueBuf = removeSpacesFromString(valueBuf); // Remove spaces from the value

    // Check if valueBuf can be converted to a valid double
    try {
        value = std::stod(valueBuf);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("Invalid numerical value in constraint definition.");
    }

    // Set class members
    constraintName = nameBuf;
    inequalityType = typeBuf;
}


void ConstraintDefinition::print(void) const{

	std::cout<<"Constraint ID = " << ID <<"\n";

	std::cout<<"Constraint definition = ";
	std::cout<< constraintName << " ";
	std::cout<< inequalityType << " ";
	std::cout<< value<< "\n";

}


std::string ConstraintDefinition::toString() const {
    std::ostringstream oss;
    oss << "Constraint ID = " << ID << "\n";
    oss << "Constraint definition = " << constraintName << " " << inequalityType << " " << value << "\n";
    return oss.str();
}


ConstraintFunction::ConstraintFunction(){}

void ConstraintFunction::setConstraintDefinition(ConstraintDefinition definitionInput){

	definitionConstraint = definitionInput;


}

void ConstraintFunction::setID(int givenID){
	definitionConstraint.ID = givenID;
}

int ConstraintFunction::getID(void) const{
	return definitionConstraint.ID;
}

void ConstraintFunction::setInequalityType(const std::string &type) {
    if (type != ">" && type != "<") {
        throw std::invalid_argument("Invalid inequality type. Expected '>' or '<'.");
    }
    definitionConstraint.inequalityType = type;
}

std::string ConstraintFunction::getInequalityType(void) const{
	return definitionConstraint.inequalityType;
}

void ConstraintFunction::setInequalityTargetValue(double value){
	definitionConstraint.value = value;

}
double ConstraintFunction::getInequalityTargetValue(void) const{
	return definitionConstraint.value;
}


void ConstraintFunction::trainSurrogate(void){

	if(!ifFunctionExplictlyDefined){
		ObjectiveFunction::trainSurrogate();
	}
}



bool ConstraintFunction::checkFeasibility(double valueIn) const{

	bool result = false;
	if (definitionConstraint.inequalityType.compare("<") == 0) {
		if (valueIn < definitionConstraint.value) {
			result = true;
		}
	}
	if (definitionConstraint.inequalityType.compare(">") == 0) {
		if (valueIn > definitionConstraint.value) {
			result = true;
		}
	}
	return result;
}


void ConstraintFunction::readOutputDesign(Design &d) const{

	vec functionalValue(1);
	functionalValue = readOutput(definition.outputFilename, 1);

	assert( int(d.constraintTrueValues.getSize()) > getID());

	d.constraintTrueValues(getID()) = functionalValue(0);

}

void ConstraintFunction::evaluateDesign(Design &d) {
    try {
        ObjectiveFunctionLogger::getInstance().log(INFO,"Evaluating the constraint with ID = " + std::to_string(definitionConstraint.ID));
        // Explicit function evaluation
        if (ifFunctionExplictlyDefined) {
            evaluateExplicitFunction(d);
        }
        // External objective function evaluation
        else {
            validateDesignParameters(d);
            setEvaluationMode("primal");

            if (!doesObjectiveFunctionPtrExist) {
                evaluateObjectiveUsingExternalExecutable(d);
            } else {
                evaluateObjectiveDirectly(d);
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error in evaluating the constraint function");
    }
}

void ConstraintFunction::evaluateExplicitFunction(Design &d) {
    vec x = d.designParameters;

    // Check if the function pointer is valid
    if (functionPtr == nullptr) {
        throw std::invalid_argument("Function pointer for the constraint is null.");
    }

    // Evaluate the constraint function
    double functionValue = functionPtr(x.getPointer());

    // Ensure the index is valid before assigning the value
    validateConstraintID(d);
    d.constraintTrueValues(getID()) = functionValue;
}

void ConstraintFunction::validateDesignParameters(const Design &d) const {
    if (d.designParameters.getSize() != dim) {
        throw std::invalid_argument("Design parameter size does not match expected dimension.");
    }
}

void ConstraintFunction::evaluateObjectiveUsingExternalExecutable(Design &d) {
    writeDesignVariablesToFile(d);

    // Execute the objective function
    evaluateObjectiveFunction();
    ObjectiveFunctionLogger::getInstance().log(INFO,"constraint evaluation is done.");


    // Read the output and assign the value
    vec functionalValue = readOutput(definition.outputFilename, 1);

    // Ensure the constraint ID is valid
    validateConstraintID(d);

    d.constraintTrueValues(getID()) = functionalValue(0);
    ObjectiveFunctionLogger::getInstance().log(INFO,"constraint value = " + std::to_string(functionalValue(0)));

}

void ConstraintFunction::evaluateObjectiveDirectly(Design &d) {
    // Directly evaluate the objective function using the design parameters
    double constraintValue = evaluateObjectiveFunctionDirectly(d.designParameters);

    // Ensure the constraint ID is valid
    validateConstraintID(d);

    d.constraintTrueValues(getID()) = constraintValue;
}

void ConstraintFunction::validateConstraintID(const Design &d) const {
    if (getID() >= static_cast<int>(d.constraintTrueValues.getSize())) {
        throw std::out_of_range("Constraint ID is out of range for constraintTrueValues.");
    }
}

void ConstraintFunction::addDesignToData(Design &d){

	assert(!definition.nameHighFidelityTrainingData.empty());
	assert(ifInitialized);

	assert(definitionConstraint.ID >= 0);

	vec newsample;
	newsample = d.constructSampleConstraint(definitionConstraint.ID);

	assert(newsample.getSize()>0);
	surrogate->addNewSampleToData(newsample);


}

double ConstraintFunction::interpolate(vec x) const{
	if(ifFunctionExplictlyDefined){

		return functionPtr(x.getPointer());
	}
	else{
		return surrogate->interpolate(x);
	}

}


pair<double, double> ConstraintFunction::interpolateWithVariance(vec x) const{


	pair<double, double> result;

	if(ifFunctionExplictlyDefined){

		result.first = functionPtr(x.getPointer());
		result.second = EPSILON;
	}
	else{
		double ftilde,sigmaSqr;
		surrogate->interpolateWithVariance(x, &ftilde, &sigmaSqr);
		result.first = ftilde;
		result.second = sqrt(sigmaSqr);
	}
	return result;
}


void ConstraintFunction::setUseExplicitFunctionOn(void){

	int ID = definitionConstraint.ID;
	assert(ID >=0 && ID<20);

	functionPtr = functionVector.at(ID);
	ifFunctionExplictlyDefined = true;


}



void ConstraintFunction::print(void) const{

	std::cout<<"\n================= Constraint function definition =========================\n";

	definitionConstraint.print();

	if(!ifFunctionExplictlyDefined){
		definition.print();
	}
	std::cout<< "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
	std::cout<<"\n==========================================================================\n";

}

std::string ConstraintFunction::toString() const {
    std::ostringstream oss;
    oss << "\n================= Constraint function definition =========================\n";
    oss << definitionConstraint.toString();  // Assuming ConstraintDefinition has a toString method

    if (!ifFunctionExplictlyDefined) {
        oss << definition.toString();  // Assuming ObjectiveFunctionDefinition has a toString method
    }

    oss << "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
    oss << "==========================================================================\n";
    return oss.str();
}



string ConstraintFunction::generateOutputString(void) const{


	std::string outputMsg;
	string tag = "Constraint function definition";
	outputMsg = generateFormattedString(tag,'=', 100) + "\n";
	outputMsg+= "Name : " + definition.name + "\n";

	if(!definition.executableName.empty()){
		outputMsg+= "Executable : " + definition.executableName + "\n";
	}

	if(doesObjectiveFunctionPtrExist){

		outputMsg+= "API Call : YES\n";
	}


	outputMsg+= "Training data : " + definition.nameHighFidelityTrainingData + "\n";

	if(!definition.outputFilename.empty()){

		outputMsg+= "Output file : " + definition.outputFilename + "\n";
	}
	if(!definition.designVectorFilename.empty()){

		outputMsg+= "Design parameters file : " + definition.designVectorFilename + "\n";
	}

	string modelName = definition.getNameOfSurrogateModel(definition.modelHiFi);
	outputMsg+= "Surrogate model : " + modelName + "\n";

	outputMsg+= "Number of iterations for model training : " +std::to_string(numberOfIterationsForSurrogateTraining) + "\n";


	std::string border(100, '=');
	outputMsg += border + "\n";


	return outputMsg;

}



bool ConstraintFunction::isUserDefinedFunction(void) const{

	return ifFunctionExplictlyDefined;
}


double ConstraintFunction::callUserDefinedFunction(vec &x) const{

	return functionPtr(x.getPointer());

}

}


