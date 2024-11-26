#include "./INCLUDE/driver_xml.hpp"
#include "./INCLUDE/driver_logger.hpp"
#include "./INCLUDE/xml_functions.hpp"
#include "./INCLUDE/string_functions.hpp"
#include "../INCLUDE/globals.hpp"

#include<cassert>
#include <map>
using namespace std;


namespace Rodop{

Driver::Driver(){}


void Driver::setConfigFileName(const std::string &filename) {
    if (filename.empty()) {
        throw std::invalid_argument("The configuration file name cannot be empty.");
    }

    configFilename = filename;
    isConfigFileSet = true;
}

void Driver::initializeKeywords(void){

	addConfigKeysGeneralSettings();
	addConfigKeysObjectiveFunction();
	addConfigKeysConstraintFunction();

	areConfigKeysInitialized = true;

}

void Driver::addConfigKey(vector<Keyword> &list, string name, string type) {

	Keyword keyToAdd;
	keyToAdd.setName(name);
	keyToAdd.setType(type);
	list.push_back(keyToAdd);

}

void Driver::addConfigKeysGeneralSettings() {
	// Clear existing configuration keys
	keywordsGeneral.clear();

	// List of keys and their corresponding types
	std::vector<std::pair<std::string, std::string>> configKeys = {
			{"name", "string"},
			{"type", "string"},
			{"display", "string"},
			{"dimension", "int"},
			{"lower_bounds", "doubleVector"},
			{"upper_bounds", "doubleVector"},
			{"number_of_function_evaluations", "int"},
			{"max_number_of_inner_iterations", "int"},
			{"number_of_threads", "int"}
	};

	// Add each key and type to the general keywords
	for (const auto& keyTypePair : configKeys) {
		addConfigKey(keywordsGeneral, keyTypePair.first, keyTypePair.second);
	}
}



SURROGATE_MODEL Driver::getSurrogateModelID(const std::string& modelName) const {
	if (modelName.empty()) {
		return ORDINARY_KRIGING;
	}


	static const std::map<std::string, SURROGATE_MODEL> modelMap = {
			{"DEFAULT", ORDINARY_KRIGING},
			{"default", ORDINARY_KRIGING},
			{"kriging", ORDINARY_KRIGING},
			{"ordinary_kriging", ORDINARY_KRIGING},
			{"universal_kriging", UNIVERSAL_KRIGING},
			{"gradient_enhanced", GRADIENT_ENHANCED},
			{"GRADIENT_ENHANCED", GRADIENT_ENHANCED},
			{"tangent_enhanced", TANGENT_ENHANCED},
			{"TANGENT_ENHANCED", TANGENT_ENHANCED}
	};

	for (const auto& pair : modelMap) {
		if (isEqual(modelName, pair.first)) {
			return pair.second;
		}
	}

	throw std::invalid_argument("Surrogate model type is unknown. Check configuration file.");
}



void Driver::setObjectiveFunctionName(ObjectiveFunctionDefinition &definition) {
	string name = getConfigKeyValueString(keywordsObjectiveFunction, "name");
	if (name.empty()) {
		throw std::invalid_argument(
				"objective function name is missing. Check configuration file");
	}
	definition.name = name;
}

void Driver::setObjectiveFunctionDesignVectorFilename(
		ObjectiveFunctionDefinition &definition) {
	string designVectorFilename = getConfigKeyValueString(
			keywordsObjectiveFunction, "design_vector_filename");
	if (designVectorFilename.empty()) {
		throw std::invalid_argument(
				"design vector filename is missing. Check configuration file");
	}
	definition.designVectorFilename = designVectorFilename;
}

void Driver::setObjectiveFunctionExecutableName(
		ObjectiveFunctionDefinition &definition) {
	string executableName = getConfigKeyValueString(keywordsObjectiveFunction,
			"executable_filename");
	if (executableName.empty()) {
		throw std::invalid_argument(
				"Executable name is missing. Check configuration file");
	}
	definition.executableName = executableName;
}

void Driver::setObjectiveFunctionTrainingDataFilename(
		ObjectiveFunctionDefinition &definition) {
	string trainingDataFilename = getConfigKeyValueString(
			keywordsObjectiveFunction, "training_data_filename");
	if (trainingDataFilename.empty()) {
		throw std::invalid_argument(
				"training data filename is missing. Check configuration file");
	}
	definition.nameHighFidelityTrainingData = trainingDataFilename;
}

void Driver::setObjectiveFunctionOutputFilename(
		ObjectiveFunctionDefinition &definition) {
	string outputDataFilename = getConfigKeyValueString(
			keywordsObjectiveFunction, "output_filename");
	if (outputDataFilename.empty()) {
		throw std::invalid_argument(
				"output data filename is missing. Check configuration file");
	}
	definition.outputFilename = outputDataFilename;
}

void Driver::setObjectiveFunctionHiFiSurrogateModelType(
		ObjectiveFunctionDefinition &definition) {
	string surrogateModelType = getConfigKeyValueString(
			keywordsObjectiveFunction, "surrogate_model_type");
	definition.modelHiFi = getSurrogateModelID(surrogateModelType);




}

void Driver::setObjectiveFunctionGradientExecutableName(
		ObjectiveFunctionDefinition &definition) {
	string executableNameGradient = getConfigKeyValueString(
			keywordsObjectiveFunction, "executable_filename_gradient");
	if (executableNameGradient.empty()) {
		throw std::invalid_argument(
				"executable name for the gradient evaluation is missing. Check configuration file");
	}
	definition.executableNameGradient = executableNameGradient;
}

void Driver::setObjectiveFunctionGradientOutputFilename(
		ObjectiveFunctionDefinition &definition) {
	string outputDataFilenameGradient = getConfigKeyValueString(
			keywordsObjectiveFunction, "output_filename_gradient");
	if (outputDataFilenameGradient.empty()) {
		throw std::invalid_argument(
				"output filename for the gradient evaluation is missing. Check configuration file");
	}
	if (isEqual(outputDataFilenameGradient, definition.outputFilename)) {
		throw std::invalid_argument(
				"output filename for the gradient evaluation is same as the function evaluation. Check configuration file");
	}
	definition.outputGradientFilename = outputDataFilenameGradient;
}

void Driver::setObjectiveFunctionDefinition(
		ObjectiveFunctionDefinition &definition) {


	setObjectiveFunctionName(definition);
	setObjectiveFunctionDesignVectorFilename(definition);
	setObjectiveFunctionExecutableName(definition);
	setObjectiveFunctionTrainingDataFilename(definition);
	setObjectiveFunctionOutputFilename(definition);

	setObjectiveFunctionHiFiSurrogateModelType(definition);

	if(definition.modelHiFi == GRADIENT_ENHANCED){

		setObjectiveFunctionGradientExecutableName(definition);
		setObjectiveFunctionGradientOutputFilename(definition);
	}



}

void Driver::setObjectiveFunctionDefinitionMultiFidelity(
		ObjectiveFunctionDefinition &definition) {
	definition.ifMultiLevel = true;
	string name = getConfigKeyValueString(keywordsObjectiveFunction, "name");
	string designVectorFilename = getConfigKeyValueString(
			keywordsObjectiveFunction, "design_vector_filename");
	if (designVectorFilename.empty()) {
		throw std::invalid_argument(
				"design vector filename is missing. Check configuration file");
	}
	definition.name = name;
	definition.designVectorFilename = designVectorFilename;
	vector<string> executableName = getConfigKeyValueStringVector(
			keywordsObjectiveFunction, "executable_filename");
	if (executableName.size() < 2) {
		throw std::invalid_argument(
				"executable names are missing. Check configuration file");
	}
	definition.executableName = executableName[0];
	definition.executableNameLowFi = executableName[1];
	vector<string> outputDataFilenames = getConfigKeyValueStringVector(
			keywordsObjectiveFunction, "output_filename");
	if (outputDataFilenames.size() < 2) {
		throw std::invalid_argument(
				"output filenames are missing. Check configuration file");
	}
	definition.outputFilename = outputDataFilenames[0];
	definition.outputFilenameLowFi = outputDataFilenames[1];
	vector<string> trainingDataFilename = getConfigKeyValueStringVector(
			keywordsObjectiveFunction, "training_data_filename");
	if (trainingDataFilename.size() < 2) {
		throw std::invalid_argument(
				"training data filenames are missing. Check configuration file");
	}
	definition.nameHighFidelityTrainingData = trainingDataFilename[0];
	definition.nameLowFidelityTrainingData = trainingDataFilename[1];
	vector<string> surrogateModelTypes = getConfigKeyValueStringVector(
			keywordsObjectiveFunction, "surrogate_model_type");
	if (surrogateModelTypes.size() < 2) {
		throw std::invalid_argument(
				"surrogate model types are missing. Check configuration file");
	}
	definition.modelHiFi = getSurrogateModelID(surrogateModelTypes[0]);
	definition.modelLowFi = getSurrogateModelID(surrogateModelTypes[1]);
	if (definition.modelHiFi == GRADIENT_ENHANCED
			&& definition.modelLowFi == GRADIENT_ENHANCED) {
		vector<string> executableNamesGradient = getConfigKeyValueStringVector(
				keywordsObjectiveFunction, "executable_filename_gradient");
		if (executableNamesGradient.size() < 2) {
			throw std::invalid_argument(
					"executable names for the gradient evaluation are missing or incomplete. Check configuration file");
		}
		definition.executableNameGradient = executableNamesGradient[0];
		definition.executableNameLowFiGradient = executableNamesGradient[1];
		vector<string> outputDataFilenamesGradient =
				getConfigKeyValueStringVector(keywordsObjectiveFunction,
						"output_filename_gradient");
		if (outputDataFilenamesGradient.size() < 2) {
			throw std::invalid_argument(
					"output file names for the gradient evaluation are missing or incomplete. Check configuration file");
		}
		definition.outputGradientFilename = outputDataFilenamesGradient[0];
		definition.outputFilenameLowFiGradient = outputDataFilenamesGradient[1];
	}
	if (definition.modelHiFi == GRADIENT_ENHANCED
			&& !(definition.modelLowFi == GRADIENT_ENHANCED)) {
		vector<string> executableNamesGradient = getConfigKeyValueStringVector(
				keywordsObjectiveFunction, "executable_filename_gradient");
		if (executableNamesGradient.size() < 1) {
			throw std::invalid_argument(
					"executable name for the gradient evaluation is missing. Check configuration file");
		}
		definition.executableNameGradient = executableNamesGradient[0];
		vector<string> outputDataFilenamesGradient =
				getConfigKeyValueStringVector(keywordsObjectiveFunction,
						"output_filename_gradient");
		definition.outputGradientFilename = outputDataFilenamesGradient[0];
	}
	if (!(definition.modelHiFi == GRADIENT_ENHANCED)
			&& definition.modelLowFi == GRADIENT_ENHANCED) {
		vector<string> executableNamesGradient = getConfigKeyValueStringVector(
				keywordsObjectiveFunction, "executable_filename_gradient");
		if (executableNamesGradient.size() < 1) {
			throw std::invalid_argument(
					"executable name for the gradient evaluation is missing. Check configuration file");
		}
		definition.executableNameLowFiGradient = executableNamesGradient[1];
		vector<string> outputDataFilenamesGradient =
				getConfigKeyValueStringVector(keywordsObjectiveFunction,
						"output_filename_gradient");
		definition.outputFilenameLowFiGradient = outputDataFilenamesGradient[1];
	}
}

void Driver::setObjectiveFunctionNumberOfTrainingIterations() {
	int numberOfTrainingIterations = getConfigKeyValueInt(
			keywordsObjectiveFunction, "number_of_training_iterations");

	if(numberOfTrainingIterations != NONEXISTINGINTKEYWORD){

		objectiveFunction.setNumberOfTrainingIterationsForSurrogateModel(
				numberOfTrainingIterations);
	}


}

void Driver::readObjectiveFunction() {


	readObjectiveFunctionKeywords();
	ObjectiveFunctionDefinition definition;


	string isMultiFidelityActive = getConfigKeyValueString(
			keywordsObjectiveFunction, "multi_fidelity");

	bool isMultiFidelity = checkIfOn(isMultiFidelityActive);


	definition.ifMultiLevel = isMultiFidelity;
	int dimension = getConfigKeyValueInt(keywordsGeneral, "dimension");
	objectiveFunction.setDimension(dimension);


	setObjectiveFunctionNumberOfTrainingIterations();

	if (!isMultiFidelity) {
		setObjectiveFunctionDefinition(definition);

	} else {

		setObjectiveFunctionDefinitionMultiFidelity(definition);
	}
	objectiveFunction.setParametersByDefinition(definition);

	assert(boxConstraints.areBoundsSet());
	objectiveFunction.setParameterBounds(boxConstraints);

	DriverLogger::getInstance().log(INFO, objectiveFunction.toString());

	optimizationStudy.addObjectFunction(objectiveFunction);

}

void Driver::setConstraintName(ConstraintDefinition &constraintDefinition) {
	string name = getConfigKeyValueString(keywordsConstraintFunction, "name");
	if (name.empty()) {
		throw std::invalid_argument(
				"Constraint name is undefined. Check configuration file");
	}
	constraintDefinition.constraintName = name;
}

void Driver::setConstraintType(ConstraintDefinition &constraintDefinition) {
	string constraint_type = getConfigKeyValueString(keywordsConstraintFunction,
			"constraint_type");
	if (constraint_type.empty()) {
		throw std::invalid_argument(
				"Constraint type is undefined. Check configuration file");
	}
	if (isEqual(constraint_type, "lt") || isEqual(constraint_type, "LT")) {
		constraintDefinition.inequalityType = "<";
	} else if (isEqual(constraint_type, "gt")
			|| isEqual(constraint_type, "TT")) {
		constraintDefinition.inequalityType = ">";
	} else {

		std::cout<<"Constraint ID = "<<constraintDefinition.ID<<"\n";
		std::cout<<"Constraint type = "<<constraint_type<<"\n";

		throw std::invalid_argument(
				"Unknown constraint type is undefined. Check configuration file");
	}

}

void Driver::setConstraintValue(ConstraintDefinition &constraintDefinition) {
	string constraint_value = getConfigKeyValueString(
			keywordsConstraintFunction, "constraint_value");
	if (constraint_value.empty()) {
		throw std::invalid_argument(
				"Constraint value is undefined. Check configuration file");
	}
	double value = 0.0;
	try {
		value = std::stod(constraint_value); // Attempt to convert string to double

	} catch (const std::invalid_argument& e) {
		throw std::invalid_argument("Invalid argument. Unable to convert string to double. Check configuration file");

	} catch (const std::out_of_range& e) {
		throw std::out_of_range("Out of range. Unable to convert string to double. Check configuration file");

	}

	constraintDefinition.value = value;

}

void Driver::setConstraintUDF(ConstraintFunction &constraint) {
	string doesUseUDF = getConfigKeyValueString(keywordsConstraintFunction,
			"user_defined_function");

	if (checkIfOn(doesUseUDF)) {
		constraint.setUseExplicitFunctionOn();
	}
}

void Driver::setConstraintNumberOfTrainingIterations(ConstraintFunction &constraint) {

	int numberOfTrainingIterations = getConfigKeyValueInt(
			keywordsObjectiveFunction, "number_of_training_iterations");

	if(numberOfTrainingIterations != -10000000){

		constraint.setNumberOfTrainingIterationsForSurrogateModel(
				numberOfTrainingIterations);

	}

}



void Driver::setConstraintDesignVectorFilename(
		ObjectiveFunctionDefinition &definition) {
	string designVectorFilename = getConfigKeyValueString(
			keywordsConstraintFunction, "design_vector_filename");
	if (designVectorFilename.empty()) {
		throw std::invalid_argument(
				"design vector filename is missing. Check configuration file");
	}
	definition.designVectorFilename = designVectorFilename;
}

void Driver::setConstraintExecutableFilename(
		ObjectiveFunctionDefinition &definition) {
	string executableName = getConfigKeyValueString(keywordsConstraintFunction,
			"executable_filename");
	if (executableName.empty()) {
		DriverLogger::getInstance().log(WARNING, "Executable name is missing. You may need to check the configuration file");
	}
	definition.executableName = executableName;
}

void Driver::setConstraintOutputFilename(
		ObjectiveFunctionDefinition &definition) {
	string outputDataFilename = getConfigKeyValueString(
			keywordsConstraintFunction, "output_filename");
	if (outputDataFilename.empty()) {
		throw std::invalid_argument(
				"output data filename is missing. Check configuration file");
	}
	definition.outputFilename = outputDataFilename;
}

void Driver::setConstraintSurrogateModelType(
		ObjectiveFunctionDefinition &definition) {
	string surrogateModelType = getConfigKeyValueString(
			keywordsConstraintFunction, "surrogate_model_type");
	definition.modelHiFi = getSurrogateModelID(surrogateModelType);
}

void Driver::setConstraintFunctionGradientExecutableName(
		ObjectiveFunctionDefinition &definition) {
	string executableNameGradient = getConfigKeyValueString(
			keywordsConstraintFunction, "executable_filename_gradient");
	if (executableNameGradient.empty()) {
		throw std::invalid_argument(
				"executable name for the constraint gradient evaluation is missing. Check configuration file");
	}
	definition.executableNameGradient = executableNameGradient;
}

void Driver::setConstraintFunctionGradientOutputFilename(
		ObjectiveFunctionDefinition &definition) {
	string outputDataFilenameGradient = getConfigKeyValueString(
			keywordsConstraintFunction, "output_filename_gradient");
	if (outputDataFilenameGradient.empty()) {
		throw std::invalid_argument(
				"output filename for the gradient evaluation is missing. Check configuration file");
	}
	if (isEqual(outputDataFilenameGradient, definition.outputFilename)) {
		throw std::invalid_argument(
				"output filename for the constraint gradient evaluation is same as the function evaluation. Check configuration file");
	}
	definition.outputGradientFilename = outputDataFilenameGradient;
}

void Driver::setConstraintFunctionExecutableFilenamesMF(
		ObjectiveFunctionDefinition &definition) {
	vector<string> executableNames = getConfigKeyValueStringVector(
			keywordsConstraintFunction, "executable_filename");
	if (executableNames.size() < 2) {
		throw std::invalid_argument(
				"executable names for the multi-fidelity constraint function are missing or incomplete. Check configuration file");
	}

	if(isEqual(executableNames[0],executableNames[1])){
		throw std::invalid_argument(
				"executable names for the multi-fidelity constraint function are same for both models. Check configuration file");

	}

	definition.executableName = executableNames[0];
	definition.executableNameLowFi = executableNames[1];

}

void Driver::setConstraintFunctionTrainingDataFilenamesMF(
		ObjectiveFunctionDefinition &definition) {
	vector<string> trainingDataFilenames = getConfigKeyValueStringVector(
			keywordsConstraintFunction, "training_data_filename");
	if (trainingDataFilenames.size() < 2) {
		throw std::invalid_argument(
				"training data filenames for the multi-fidelity constraint function are missing or incomplete. Check configuration file");
	}

	if(isEqual(trainingDataFilenames[0],trainingDataFilenames[1])){
		throw std::invalid_argument(
				"training data filenames for the multi-fidelity constraint function are same for both models. Check configuration file");

	}

	definition.nameHighFidelityTrainingData = trainingDataFilenames[0];
	definition.nameLowFidelityTrainingData  = trainingDataFilenames[1];

}

void Driver::setConstraintFunctionOutputFilenamesMF(
		ObjectiveFunctionDefinition &definition) {
	vector<string> outputFilenames = getConfigKeyValueStringVector(
			keywordsConstraintFunction, "output_filename");
	if (outputFilenames.size() < 2) {
		throw std::invalid_argument(
				"output filenames for the multi-fidelity constraint function are missing or incomplete. Check configuration file");
	}

	if(isEqual(outputFilenames[0],outputFilenames[1])){
		throw std::invalid_argument(
				"output filenames for the multi-fidelity constraint function are same for both models. Check configuration file");

	}

	definition.outputFilename = outputFilenames[0];
	definition.outputFilenameLowFi  = outputFilenames[1];

}

void Driver::setConstraintFunctionSurrogateTypesMF(
		ObjectiveFunctionDefinition &definition) {
	vector<string> surrogateModelTypes = getConfigKeyValueStringVector(
			keywordsConstraintFunction, "surrogate_model_type");

	definition.modelHiFi = getSurrogateModelID(surrogateModelTypes[0]);
	definition.modelLowFi = getSurrogateModelID(surrogateModelTypes[1]);

}

void Driver::setConstrantFunctionTrainingDataFilename(
		ObjectiveFunctionDefinition &definition) {
	string trainingDataFilename = getConfigKeyValueString(
			keywordsConstraintFunction, "training_data_filename");
	if (trainingDataFilename.empty()) {
		throw std::invalid_argument(
				"training data filename is missing. Check configuration file");
	}
	definition.nameHighFidelityTrainingData = trainingDataFilename;
}

void Driver::readConstraintFunctionsSingleFidelity(
		ObjectiveFunctionDefinition &definition) {
	setConstraintExecutableFilename(definition);
	setConstraintDesignVectorFilename(definition);
	setConstraintOutputFilename(definition);
	setConstrantFunctionTrainingDataFilename(definition);
	setConstraintSurrogateModelType(definition);
	if (definition.modelHiFi == GRADIENT_ENHANCED) {
		setConstraintFunctionGradientExecutableName(definition);
		setConstraintFunctionGradientOutputFilename(definition);
	}
}

void Driver::readConstraintFunctionsMultiFidelity(
		ObjectiveFunctionDefinition &definition) {
	setConstraintFunctionExecutableFilenamesMF(definition);
	setConstraintFunctionTrainingDataFilenamesMF(definition);
	setConstraintFunctionOutputFilenamesMF(definition);
	setConstraintFunctionSurrogateTypesMF(definition);
}

void Driver::readConstraintFunctions(void) {


	vector<string> constraintDefinitionsText = readConstraintFunctionsFromXML();


	int ID = 0;
	for (vector<string>::iterator i = constraintDefinitionsText.begin(); i != constraintDefinitionsText.end(); ++i){



		string constraintFunctionText = *i;

		//		std::cout<<constraintFunctionText<<"\n";
		/* we check if multilevel model is active */

		readConstraintFunctionKeywords( constraintFunctionText);

		ConstraintDefinition constraintDefinition;
		ObjectiveFunctionDefinition definition;
		ConstraintFunction constraint;

		int dimension = getConfigKeyValueInt(keywordsGeneral, "dimension");
		constraint.setDimension(dimension);

		constraintDefinition.ID = ID;
		constraint.setID(ID);

		setConstraintNumberOfTrainingIterations(constraint);


		string isMultiFidelityActive = getConfigKeyValueString(
				keywordsConstraintFunction, "multi_fidelity");
		bool isMultiFidelity = checkIfOn(isMultiFidelityActive);
		definition.ifMultiLevel = isMultiFidelity;


		setConstraintName(constraintDefinition);
		definition.name = constraintDefinition.constraintName;

		setConstraintType(constraintDefinition);
		setConstraintValue(constraintDefinition);
		setConstraintUDF(constraint);

		if(!isMultiFidelity){

			/* if the constraint is not a user defined function, it will require more information like name of the training data etc */
			if(!constraint.isUserDefinedFunction()){

				readConstraintFunctionsSingleFidelity(definition);
			}

		}
		else{

			readConstraintFunctionsMultiFidelity(definition);

			if (definition.modelHiFi == GRADIENT_ENHANCED
					&& definition.modelLowFi == GRADIENT_ENHANCED) {
				vector<string> executableNamesGradient = getConfigKeyValueStringVector(
						keywordsConstraintFunction, "executable_filename_gradient");
				if (executableNamesGradient.size() < 2) {
					throw std::invalid_argument(
							"executable names for the gradient evaluation are missing or incomplete. Check configuration file");
				}
				definition.executableNameGradient = executableNamesGradient[0];
				definition.executableNameLowFiGradient = executableNamesGradient[1];


				vector<string> outputDataFilenamesGradient =
						getConfigKeyValueStringVector(keywordsConstraintFunction,
								"output_filename_gradient");
				if (outputDataFilenamesGradient.size() < 2) {
					throw std::invalid_argument(
							"output file names for the gradient evaluation are missing or incomplete. Check configuration file");
				}
				definition.outputGradientFilename = outputDataFilenamesGradient[0];
				definition.outputFilenameLowFiGradient = outputDataFilenamesGradient[1];
			}

			if (!(definition.modelHiFi == GRADIENT_ENHANCED)
					&& definition.modelLowFi == GRADIENT_ENHANCED) {
				vector<string> executableNamesGradient = getConfigKeyValueStringVector(
						keywordsConstraintFunction, "executable_filename_gradient");
				if (executableNamesGradient.size() < 1) {
					throw std::invalid_argument(
							"executable names for the gradient evaluation are missing or incomplete. Check configuration file");
				}

				definition.executableNameLowFiGradient = executableNamesGradient[0];
				vector<string> outputDataFilenamesGradient =
						getConfigKeyValueStringVector(keywordsConstraintFunction,
								"output_filename_gradient");
				if (outputDataFilenamesGradient.size() < 1) {
					throw std::invalid_argument(
							"output file names for the gradient evaluation are missing or incomplete. Check configuration file");
				}
				definition.outputFilenameLowFiGradient = outputDataFilenamesGradient[0];
			}


			if (definition.modelHiFi == GRADIENT_ENHANCED
					&& !(definition.modelLowFi == GRADIENT_ENHANCED)) {
				vector<string> executableNamesGradient = getConfigKeyValueStringVector(
						keywordsConstraintFunction, "executable_filename_gradient");
				if (executableNamesGradient.size() < 1) {
					throw std::invalid_argument(
							"executable names for the gradient evaluation are missing or incomplete. Check configuration file");
				}

				definition.executableNameGradient = executableNamesGradient[0];
				vector<string> outputDataFilenamesGradient =
						getConfigKeyValueStringVector(keywordsConstraintFunction,
								"output_filename_gradient");
				if (outputDataFilenamesGradient.size() < 1) {
					throw std::invalid_argument(
							"output file names for the gradient evaluation are missing or incomplete. Check configuration file");
				}
				definition.outputFilename = outputDataFilenamesGradient[0];
			}

		}


		constraint.setParametersByDefinition(definition);
		constraint.setConstraintDefinition(constraintDefinition);

		assert(boxConstraints.areBoundsSet());
		constraint.setParameterBounds(boxConstraints);

//		constraint.print();

		optimizationStudy.addConstraint(constraint);

		ID++;

	}

}
void Driver::readOptimizationParameters() {
	// Log that we're starting the process
	DriverLogger::getInstance().log(INFO, "Starting to read optimization parameters.");

	std::string parameters_section = readASegmentFromXMLFile("optimization_parameters");

	if (parameters_section.empty()) {
		// Log if the section is empty
		DriverLogger::getInstance().log(WARNING, "No 'optimization_parameters' section found in the XML file.");
		return;
	}

	// Log that parameters were found
	DriverLogger::getInstance().log(INFO, "Extracting parameters from 'optimization_parameters' section.");

	std::vector<std::string> parameters = extractContents(parameters_section, "parameter");

	for (const auto &param : parameters) {
		std::string name = getStringValueFromXML(param, "name");
		std::string type = getStringValueFromXML(param, "type");

		// Ensure name and type are not empty
		if (name.empty()) {
			DriverLogger::getInstance().log(ERROR, "Parameter 'name' is missing or empty.");
			throw std::runtime_error("Driver::readOptimizationParameters: 'name' cannot be empty.");
		}
		if (type.empty()) {
			DriverLogger::getInstance().log(ERROR, "Parameter 'type' is missing or empty for parameter: " + name);
			throw std::runtime_error("Driver::readOptimizationParameters: 'type' cannot be empty.");
		}

		// Convert the type to lowercase for case-insensitive comparison
		type = toLower(type);

		// Validate type to ensure it's either "continuous", "discrete", "real", or "int"
		if (type != "continuous" && type != "discrete" && type != "real" && type != "int") {
			DriverLogger::getInstance().log(ERROR, "Invalid parameter type '" + type + "' for parameter: " + name + ". Must be 'continuous', 'discrete', 'real', or 'int'.");
			throw std::runtime_error("Driver::readOptimizationParameters: Invalid type '" + type + "' for parameter: " + name);
		}

		// Log the parameter being processed
		DriverLogger::getInstance().log(INFO, "Processing parameter: " + name + " of type: " + type);

		double lb = getDoubleValueFromXML(param, "lower_bound");
		double ub = getDoubleValueFromXML(param, "upper_bound");

		// Check that lower_bound is less than or equal to upper_bound
		if (lb > ub) {
			DriverLogger::getInstance().log(ERROR, "Invalid bounds for parameter '" + name + "': lower_bound > upper_bound.");
			throw std::runtime_error("Driver::readOptimizationParameters: 'lower_bound' > 'upper_bound' for parameter: " + name);
		}

		double inc = getDoubleValueFromXML(param, "increment");

		// Special handling for discrete and int types
		if (type == "discrete" || type == "int") {
			// Ensure the increment is defined, greater than 0, and less than or equal to ub - lb
			if (inc <= 0 || inc > (ub - lb)) {
				DriverLogger::getInstance().log(ERROR, "Invalid increment for 'discrete' or 'int' parameter '" + name + "': increment must be > 0 and <= (upper_bound - lower_bound).");
				throw std::runtime_error("Driver::readOptimizationParameters: Invalid increment for 'discrete' or 'int' parameter: " + name);
			}
		} else {
			// For continuous and real types, increment is zero
			inc = 0.0;
		}

		// Create and store the parameter
		Parameter newParameter {name, type, lb, ub, inc};
		designParameters.push_back(newParameter);

		// Log that the parameter was successfully added
		DriverLogger::getInstance().log(INFO, "Added parameter: " + name + " with bounds [" + std::to_string(lb) + ", " + std::to_string(ub) + "] and increment " + std::to_string(inc));
	}

	// Log completion of the parameter reading
	DriverLogger::getInstance().log(INFO, "Completed reading optimization parameters.");
}



void Driver::readConfigurationFile(void){


	assert(isConfigFileSet);

	initializeKeywords();

	readGeneralSettings();


	if(isOptimization){
		readOptimizationParameters();
		setBoxConstraints();
		readObjectiveFunction();
		readConstraintFunctions();
		setBoxConstraintsOptimizationStudy();
		setParameterNamesOptimizationStudy();
		setDiscreteParametersOptimizationStudy();

	}


	isConfigurationFileRead = true;



}

void Driver::setDimensionOptimizationStudy(void) {
	int dimension = getConfigKeyValueInt(keywordsGeneral, "dimension");
	if (dimension <= 0) {
		throw std::invalid_argument(
				"dimension must be a positive integer. Check configuration file");
	}
	optimizationStudy.setDimension(dimension);
}

void Driver::setNumberOfFunctionEvaluationsOptimizationStudy(void) {
	int N = getConfigKeyValueInt(keywordsGeneral, "number_of_function_evaluations");
	if (N <= 0) {

		throw std::invalid_argument(
				"number of function evaluations must be a positive integer. Check configuration file");
	}
	optimizationStudy.setMaximumNumberOfIterations(N);
}

void Driver::setMaxNumberOfInnerIterationsOptimizationStudy(void) {
	int maxiter = getConfigKeyValueInt(keywordsGeneral, "max_number_of_inner_iterations");

	if(maxiter != -10000000){
		optimizationStudy.setMaximumNumberOfInnerIterations(maxiter);
	}


}

void Driver::setNumberOfThreadsOptimizationStudy(void) {

	int N = getConfigKeyValueInt(keywordsGeneral, "number_of_threads");

	if(N != -10000000){

		optimizationStudy.setNumberOfThreads(N);
	}

}

void Driver::setNameOptimizationStudy() {
	string name = getConfigKeyValueString(keywordsGeneral, "name");
	if (name.empty()) {
		optimizationStudy.setName(name);
	} else {
		optimizationStudy.setName("OptimizationStudy");
	}
}


void Driver::setParameterNamesOptimizationStudy() {

	vector<string> names;

	for (size_t i = 0; i < designParameters.size(); ++i) {

		if(!designParameters[i].name.empty()){

			names.push_back(designParameters[i].name);
		}

	}

	if(!names.empty()){
		optimizationStudy.setParameterNames(names);
	}


}


void Driver::setDiscreteParametersOptimizationStudy() {

	int dimension = getConfigKeyValueInt(keywordsGeneral, "dimension");
	if (dimension != static_cast<int>(designParameters.size())) {
		std::cout<<"Dimension = " << dimension <<"\n";
		std::cout<<"Number of design parameters = " << static_cast<int>(designParameters.size()) <<"\n";
		throw std::invalid_argument("Mismatch between 'dimension' and the number of parameters. Check the configuration file.");
	}

	// Populate lb and ub from designParameters.
	for (size_t i = 0; i < designParameters.size(); ++i) {

		if(designParameters[i].type == "discrete" || designParameters[i].type == "int"){

			int index = int(i);
			optimizationStudy.setParameterToDiscrete(index, designParameters[i].increment);
		}

	}

}

void Driver::setBoxConstraints() {
	// Ensure that keywordsGeneral is not empty before proceeding.
	assert(!keywordsGeneral.empty());

	// Get the 'dimension' from configuration and check against the size of designParameters.
	int dimension = getConfigKeyValueInt(keywordsGeneral, "dimension");
	if (dimension != static_cast<int>(designParameters.size())) {
		std::cout<<"Dimension = " << dimension <<"\n";
		std::cout<<"Number of design parameters = " << static_cast<int>(designParameters.size()) <<"\n";
		throw std::invalid_argument("Mismatch between 'dimension' and the number of parameters. Check the configuration file.");
	}

	// Initialize lower bounds (lb) and upper bounds (ub) vectors.
	std::vector<double> lb(designParameters.size());
	std::vector<double> ub(designParameters.size());

	// Populate lb and ub from designParameters.
	for (size_t i = 0; i < designParameters.size(); ++i) {
		lb[i] = designParameters[i].lb;
		ub[i] = designParameters[i].ub;
	}

	// Set bounds for boxConstraints using lb and ub vectors.
	boxConstraints.setBounds(lb, ub);
}


void Driver::setBoxConstraintsOptimizationStudy() {

	optimizationStudy.setBoxConstraints(boxConstraints);
}




void Driver::readGeneralSettings(void) {
	// Read the general settings segment from the XML file
	std::string input = readASegmentFromXMLFile("general_settings");

	// Update values of keywordsGeneral from the XML input
	for (auto& keyword : keywordsGeneral) {
		keyword.getValueFromXMLString(input);
		// keyword.print(); // Uncomment if needed
	}

	// Get the value of 'type' from the configuration
	std::string type = getConfigKeyValueString(keywordsGeneral, "type");

	// Ensure 'type' is defined
	if (type.empty()) {
		throw std::invalid_argument("Type must be defined. Check configuration file.");
	}

	// Check if the type indicates an optimization study
	if (isEqual(type, "optimization")) {
		isOptimization = true;
	}

	// Execute optimization study-specific settings if it's an optimization study
	if (isOptimization) {
		setDimensionOptimizationStudy();
		setNameOptimizationStudy();
		setNumberOfFunctionEvaluationsOptimizationStudy();
		setMaxNumberOfInnerIterationsOptimizationStudy();
		setNumberOfThreadsOptimizationStudy();
	}
}


void Driver::addConfigKeysConstraintFunction() {
	// Clear the existing keys
	keywordsConstraintFunction.clear();

	// Define a list of keys with their associated types
	std::vector<std::pair<std::string, std::string>> configKeys = {
			{"multi_fidelity", "string"},
			{"name", "string"},
			{"constraint_value", "string"},
			{"constraint_type", "string"},
			{"user_defined_function", "string"},
			{"design_vector_filename", "string"},
			{"warm_start", "string"},
			{"output_filename", "string"},
			{"output_filename_gradient", "string"},
			{"executable_filename", "string"},
			{"executable_filename_gradient", "string"},
			{"executable_filename_tangent", "string"},
			{"surrogate_model_type", "string"},
			{"training_data_filename", "string"},
			{"number_of_training_iterations", "int"}
	};

	// Iterate over the vector and add configuration keys
	for (const auto& key : configKeys) {
		addConfigKey(keywordsConstraintFunction, key.first, key.second);
	}
}


void Driver::addConfigKeysConstraintFunctionMultiFidelity(void) {
	keywordsConstraintFunction.clear();

	// List of single value keys with their associated types
	std::vector<std::pair<std::string, std::string>> singleValueKeys = {
			{"multi_fidelity", "string"},
			{"name", "string"},
			{"constraint_value", "string"},
			{"constraint_type", "string"},
			{"user_defined_function", "string"},
			{"design_vector_filename", "string"},
			{"warm_start", "string"},
			{"number_of_training_iterations", "int"}
	};

	// List of vector value keys with their associated types
	std::vector<std::pair<std::string, std::string>> vectorValueKeys = {
			{"output_filename", "stringVector"},
			{"output_filename_gradient", "stringVector"},
			{"executable_filename", "stringVector"},
			{"executable_filename_gradient", "stringVector"},
			{"executable_filename_tangent", "stringVector"},
			{"surrogate_model_type", "stringVector"},
			{"training_data_filename", "stringVector"}
	};

	// Add single value keys
	for (const auto& keyType : singleValueKeys) {
		addConfigKey(keywordsConstraintFunction, keyType.first, keyType.second);
	}

	// Add vector value keys
	for (const auto& keyType : vectorValueKeys) {
		addConfigKey(keywordsConstraintFunction, keyType.first, keyType.second);
	}
}


void Driver::addConfigKeysObjectiveFunction(void) {

	keywordsObjectiveFunction.clear();



	addConfigKey(keywordsObjectiveFunction, "multi_fidelity", "string");
	addConfigKey(keywordsObjectiveFunction, "name", "string");
	addConfigKey(keywordsObjectiveFunction, "design_vector_filename", "string");
	addConfigKey(keywordsObjectiveFunction, "warm_start", "string");

	addConfigKey(keywordsObjectiveFunction, "output_filename", "string");
	addConfigKey(keywordsObjectiveFunction, "output_filename_gradient", "string");

	addConfigKey(keywordsObjectiveFunction, "executable_filename", "string");
	addConfigKey(keywordsObjectiveFunction, "executable_filename_gradient", "string");
	addConfigKey(keywordsObjectiveFunction, "executable_filename_tangent", "string");
	addConfigKey(keywordsObjectiveFunction, "surrogate_model_type", "string");
	addConfigKey(keywordsObjectiveFunction, "training_data_filename", "string");
	addConfigKey(keywordsObjectiveFunction, "number_of_training_iterations", "int");


}

void Driver::addConfigKeysObjectiveFunctionMultiFidelity(void) {

	keywordsObjectiveFunction.clear();



	addConfigKey(keywordsObjectiveFunction, "multi_fidelity", "string");
	addConfigKey(keywordsObjectiveFunction, "name", "string");
	addConfigKey(keywordsObjectiveFunction, "design_vector_filename", "string");
	addConfigKey(keywordsObjectiveFunction, "warm_start", "string");

	addConfigKey(keywordsObjectiveFunction, "output_filename", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "output_filename_gradient", "stringVector");

	addConfigKey(keywordsObjectiveFunction, "executable_filename", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "executable_filename_gradient", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "executable_filename_tangent", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "surrogate_model_type", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "training_data_filename", "stringVector");
	addConfigKey(keywordsObjectiveFunction, "number_of_training_iterations", "int");


}


int Driver::getConfigKeyValueInt(vector<Keyword>& list, const string& key) {
	if (list.empty()) {
		throw std::invalid_argument("Keyword list is empty.");
	}
	if (key.empty()) {
		throw std::invalid_argument("Key is empty.");
	}
	// Loop through the list using a range-based for loop and auto
	for (const auto& keyword : list) {
		if (keyword.getName() == key) {
			return keyword.getIntValue();
		}
	}

	return NONEXISTINGINTKEYWORD;
}


double Driver::getConfigKeyValueDouble(vector<Keyword>& list, const string& key) {

	if (list.empty()) {
		throw std::invalid_argument("Keyword list is empty.");
	}
	if (key.empty()) {
		throw std::invalid_argument("Key is empty.");
	}

	double returnValue = EPSILON;

	// Using a range-based for loop with const reference to avoid unnecessary copies
	for (const auto& keyword : list) {
		if (keyword.getName() == key) {
			returnValue = keyword.getDoubleValue();
			break;
		}
	}

	return returnValue;
}


vector<double> Driver::getConfigKeyValueDoubleVector(vector<Keyword>& list, const string& key) {
	if (list.empty()) {
		throw std::invalid_argument("Keyword list is empty.");
	}
	if (key.empty()) {
		throw std::invalid_argument("Key is empty.");
	}

	vector<double> result;  // Starts empty
	for (const auto& keyword : list) {
		if (keyword.getName() == key) {
			result = keyword.getDoubleValueVector();
			break;
		}
	}

	return result;
}



string Driver::getConfigKeyValueString(vector<Keyword> &list, const string &key) {
    if (list.empty()) {
        DriverLogger::getInstance().log(ERROR, "Keyword list is empty.");
        throw std::invalid_argument("Keyword list is empty.");
    }
    if (key.empty()) {
        DriverLogger::getInstance().log(ERROR, "Provided key is empty.");
        throw std::invalid_argument("Key is empty.");
    }

    // Log that we're searching for the key
    DriverLogger::getInstance().log(INFO, "Searching for key: " + key);

    for (const auto& keyword : list) {
        if (keyword.getName() == key) {
            DriverLogger::getInstance().log(INFO, "Found key: " + key + ", returning value.");
            return keyword.getStringValue();
        }
    }

    // If key was not found, log and return an empty string
    DriverLogger::getInstance().log(WARNING, "Key '" + key + "' not found in the keyword list.");
    return "";
}


vector<string> Driver::getConfigKeyValueStringVector(vector<Keyword>& list, const string& key) {
	if (list.empty()) {
		throw std::invalid_argument("Keyword list is empty.");
	}
	if (key.empty()) {
		throw std::invalid_argument("Key is empty.");
	}
	vector<string> returnValue;  // Defaults to an empty vector
	for (const auto& keyword : list) {
		if (keyword.getName() == key) {
			returnValue = keyword.getStringValueVector();
			break;
		}
	}

	return returnValue;
}




std::string Driver::getConfigKeyValueString(vector<Keyword> &list, const std::string &xml_input, const std::string &key) {

    // Check for an empty list and log an error if applicable
    if (list.empty()) {
        DriverLogger::getInstance().log(ERROR, "Keyword list is empty.");
        throw std::invalid_argument("List is empty.");
    }

    // Check for an empty XML input and log an error if applicable
    if (xml_input.empty()) {
        DriverLogger::getInstance().log(ERROR, "XML input is empty.");
        throw std::invalid_argument("XML input is empty.");
    }

    // Check for an empty key and log an error if applicable
    if (key.empty()) {
        DriverLogger::getInstance().log(ERROR, "Key is empty.");
        throw std::invalid_argument("Key is empty.");
    }

    // Log that we're starting to search for the key
    DriverLogger::getInstance().log(INFO, "Searching for key: " + key);

    std::string returnString;
    for (auto &keyword : list) {
        if (keyword.getName() == key) {
            // Log that the key was found
            DriverLogger::getInstance().log(INFO, "Key '" + key + "' found, extracting value from XML.");

            // Extract the value from the XML input
            keyword.getValueFromXMLString(xml_input);
            returnString = keyword.getStringValue();
            break;
        }
    }

    // If the key was not found in the list, log a warning
    if (returnString.empty()) {
        DriverLogger::getInstance().log(WARNING, "Key '" + key + "' not found in the keyword list.");
    }

    return returnString;
}

void Driver::readObjectiveFunctionKeywords() {
    // Check if configuration keys are initialized, if not, throw an exception
    if (!areConfigKeysInitialized) {
        DriverLogger::getInstance().log(ERROR, "Configuration keys are not initialized.");
        throw std::runtime_error("Configuration keys are not initialized.");
    }

    // Check if the configuration file is set, if not, throw an exception
    if (!isConfigFileSet) {
        DriverLogger::getInstance().log(ERROR, "Configuration file is not set.");
        throw std::runtime_error("Configuration file is not set.");
    }

    // Log that we're starting to read the objective function keywords
    DriverLogger::getInstance().log(INFO, "Starting to read objective function keywords from the XML file.");

    // Read the objective function segment from the XML file
    std::string objectiveFunctionText = readASegmentFromXMLFile("objective_function");
    DriverLogger::getInstance().log(INFO, "Objective function segment read successfully from XML.");

    // Check if the multi-fidelity model is active
    std::string isMultiFidelityActive = getConfigKeyValueString(keywordsObjectiveFunction, objectiveFunctionText, "multi_fidelity");
    bool isMultiFidelity = checkIfOn(isMultiFidelityActive);

    if (isMultiFidelity) {
        // Log that multi-fidelity is active
        DriverLogger::getInstance().log(INFO, "Multi-fidelity model is active. Adding multi-fidelity configuration keys.");

        // Add multi-fidelity configuration keys
        addConfigKeysObjectiveFunctionMultiFidelity();
    } else {
        // Log that multi-fidelity is inactive
        DriverLogger::getInstance().log(INFO, "Multi-fidelity model is not active.");
    }

    // Loop through the objective function keywords and extract values from the XML string
    for (auto &keyword : keywordsObjectiveFunction) {
        // Log the keyword being processed
        DriverLogger::getInstance().log(INFO, "Processing keyword: " + keyword.getName());

        // Extract the keyword value from the XML string
        try {
            keyword.getValueFromXMLString(objectiveFunctionText);
            DriverLogger::getInstance().log(INFO, "Successfully extracted value for keyword: " + keyword.getName());
        } catch (const std::exception &e) {
            // Log any errors encountered while extracting the value
            DriverLogger::getInstance().log(ERROR, "Error extracting value for keyword: " + keyword.getName() + " - " + e.what());
            throw;
        }

        keyword.printToLog();
    }

    // Log completion of reading the objective function keywords
    DriverLogger::getInstance().log(INFO, "Completed reading objective function keywords.");
}




void Driver::readConstraintFunctionKeywords(string inputText) {

	assert(!inputText.empty());

	string isMultiFidelityActive = getConfigKeyValueString(keywordsConstraintFunction, inputText ,"multi_fidelity" );

	bool isMultiFidelity = checkIfOn(isMultiFidelityActive);

	if(isMultiFidelity){
		addConfigKeysConstraintFunctionMultiFidelity();
	}
	else{
		addConfigKeysConstraintFunction();
	}

	for (std::vector<Keyword>::iterator i = keywordsConstraintFunction.begin(); i != keywordsConstraintFunction.end(); ++i){
		i->getValueFromXMLString(inputText);
		//		i->print();

	}

}



std::string Driver::readASegmentFromXMLFile(string keyword) const {

	assert(!keyword.empty());
	assert(!configFilename.empty());

	std::ifstream file(configFilename);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + configFilename);
	}

	std::stringstream buffer;
	buffer << file.rdbuf(); // Read the entire file into the stringstream

	std::string filetext =  buffer.str(); // Convert stringstream to std::string and return


	return extractContentBetweenKeys(filetext, keyword);

}



std::vector<std::string> Driver::readConstraintFunctionsFromXML(void) const {
	std::vector<std::string> constraintFunctions;
	std::ifstream file(configFilename);
	std::string line;
	bool insideConstraintFunction = false;
	std::string currentConstraintFunction;

	// Check if the file was opened successfully
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << configFilename << std::endl;
		return constraintFunctions;
	}

	// Read the file line by line
	while (std::getline(file, line)) {
		// Trim leading and trailing whitespaces from the line (if needed)
		line.erase(0, line.find_first_not_of(" \t"));  // Left trim
		line.erase(line.find_last_not_of(" \t") + 1);  // Right trim

		// Start of a constraint function
		if (line.find("<constraint_function>") != std::string::npos) {
			insideConstraintFunction = true;
			currentConstraintFunction.clear();
		}

		// End of a constraint function
		if (line.find("</constraint_function>") != std::string::npos) {
			insideConstraintFunction = false;
			constraintFunctions.push_back(currentConstraintFunction);
		}

		// Collect lines inside the constraint function block
		if (insideConstraintFunction) {
			currentConstraintFunction += line + "\n";  // Preserve line breaks
		}
	}

	file.close();  // Close the file when done
	return constraintFunctions;
}

void Driver::runOptimization(void) {

    if (!isOptimization) {
        throw std::runtime_error("Optimization flag is not set. Cannot run optimization.");
    }

    if (!isConfigurationFileRead) {
        throw std::runtime_error("Configuration file has not been read. Cannot run optimization.");
    }

    // Perform the optimization study
    optimizationStudy.performEfficientGlobalOptimization();
}

void Driver::run(void) {

    // Check if optimization flag is set
    if (!isOptimization) {
        throw std::runtime_error("Cannot run: Optimization flag is not set.");
    }

    // Check if the configuration file has been read
    if (!isConfigurationFileRead) {
        throw std::runtime_error("Cannot run: Configuration file has not been read.");
    }

    // Run the optimization if all checks pass
    runOptimization();
}



} /* Namespace Rodop */

