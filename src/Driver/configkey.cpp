#include <string>
#include <fstream>
#include <sstream>
#include "./INCLUDE/configkey.hpp"
#include "./INCLUDE/string_functions.hpp"
#include "./INCLUDE/driver_logger.hpp"
#include "./INCLUDE/xml_functions.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"

namespace Rodop{

void Keyword::returnErrorIfNotSet(void) const{

	if(!isValueSet){

		std::string msg = name + " is not set! Check the configuration file.";
		throw std::runtime_error("Error:" + msg);
	}
}

void Keyword::setName(string input){

	if(input.empty()){

		throw std::runtime_error("Error: empty name in Keyword::setName.");
	}

	name = input;


}

string Keyword::getName(void) const {
	return name;
}

string Keyword::getStringValue(void) const{

	if(type == "string"){
		return valueString;
	}
	else{

		throw std::invalid_argument("keyword: " + name +  " is not a string");
	}
}


vector<string> Keyword::getStringValueVector(void) const{

	if(type == "stringVector"){
		return valueStringVector;
	}
	else{

		throw std::invalid_argument("keyword: " + name +  " is not a string vector");
	}
}


string Keyword::getStringValueVectorAt(unsigned indx) const{

	if(indx >=  valueStringVector.size()){

		throw std::invalid_argument("index is beyond valueStringVector.size()");
	}

	return valueStringVector[indx];
}

double Keyword::getDoubleValueVectorAt(unsigned indx) const{

	if(indx >=  valueDoubleVector.size()){

		throw std::invalid_argument("index is beyond valueDoubleVector.size()");
	}

	return valueDoubleVector[indx];
}

double Keyword::getDoubleValue(void) const{

	if(type == "double"){
		return valueDouble;
	}
	else{
		throw std::invalid_argument("keyword is not a double");
	}
}

vector<double> Keyword::getDoubleValueVector(void) const{

	if(type == "doubleVector"){
		return valueDoubleVector;
	}
	else{
		throw std::invalid_argument("keyword is not a double vector");
	}
}

int Keyword::getIntValue(void) const{

	if(type == "int"){
		return valueInt;
	}
	else{
		throw std::invalid_argument("keyword is not an integer");
	}

}

void Keyword::setType(string whichType){

	if(!(whichType == "string" || whichType == "int" || whichType == "double" || whichType == "intVector"
			|| whichType == "doubleVector" || whichType == "stringVector")){

		std::string msg = whichType + " is not a valid type.";
		throw std::runtime_error("Error:" + msg);
	}
	else{
		type = whichType;
	}
}

void Keyword::getValueFromXMLString(const std::string& input) {
    if (name.empty()) {
        throw std::runtime_error("Error: Keyword name is empty.");
    }

    // Set flag for whether the value was successfully extracted
    isValueSet = false;

    // Use a helper lambda for setting values and checking for success
    auto setValue = [&](auto& destination, const auto& source) {
        if (!source.empty()) {
            destination = source;
            isValueSet = true;
        }
    };

    // Determine type and extract the corresponding value
    if (type == "string") {
        setValue(valueString, getStringValueFromXML(input, name));
    }
    else if (type == "stringVector") {
        setValue(valueStringVector, getStringVectorValuesFromXML(input, name));
    }
    else if (type == "double") {
        double value = getDoubleValueFromXML(input, name);
        if (std::fabs(value) > std::numeric_limits<double>::epsilon()) {
            valueDouble = value;
            isValueSet = true;
        }
    }
    else if (type == "doubleVector") {
        setValue(valueDoubleVector, getDoubleVectorValuesFromXML(input, name));
    }
    else if (type == "int") {
        int value = getIntegerValueFromXML(input, name);
        if (value != -10000000) {  // Assuming -10000000 is an error code
            valueInt = value;
            isValueSet = true;
        }
    }
    else {
        throw std::invalid_argument("Error: Unsupported keyword type for " + name);
    }
}

void Keyword::print(void) const {
    std::cout << name;

    if (isValueSet) {
        std::cout << " : ";
        if (type == "string") {
            std::cout << valueString << "\n";
        } else if (type == "double") {
            std::cout << valueDouble << "\n";
        } else if (type == "int") {
            std::cout << valueInt << "\n";
        } else if (type == "doubleVector") {
            for (const auto& element : valueDoubleVector) {
                std::cout << element << std::endl;
            }
        } else if (type == "stringVector") {
            for (const auto& element : valueStringVector) {
                std::cout << element << std::endl;
            }
        }
    } else {
        std::cout << "\n";
    }
}


void Keyword::printToLog() const {
    std::ostringstream logStream;
    logStream << name;

    if (isValueSet) {
        logStream << " : ";
        if (type == "string") {
            logStream << valueString;
        } else if (type == "double") {
            logStream << valueDouble;
        } else if (type == "int") {
            logStream << valueInt;
        } else if (type == "doubleVector") {
            logStream << "Double Vector: [";
            for (size_t i = 0; i < valueDoubleVector.size(); ++i) {
                logStream << valueDoubleVector[i];
                if (i < valueDoubleVector.size() - 1) logStream << ", ";
            }
            logStream << "]";
        } else if (type == "stringVector") {
            logStream << "String Vector: [";
            for (size_t i = 0; i < valueStringVector.size(); ++i) {
                logStream << valueStringVector[i];
                if (i < valueStringVector.size() - 1) logStream << ", ";
            }
            logStream << "]";
        }
    } else {
        logStream << " (Value not set)";
    }

    // Log the final message
    DriverLogger::getInstance().log(INFO, logStream.str());
}

} /* Namespace Rodop */

