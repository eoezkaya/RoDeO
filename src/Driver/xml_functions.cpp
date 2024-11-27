#include "./INCLUDE/xml_functions.hpp"
#include "./INCLUDE/string_functions.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"

#include <sstream>

namespace Rodop{

template <typename T>
std::string generateXml(const std::string& elementName, const T& value) {
	std::ostringstream xml;

	// Apply the std::fixed manipulator before inserting values
	xml << std::fixed;
	xml << "<" << elementName << ">" << value << "</" << elementName << ">";

	return xml.str();
}


//template <typename T>
//string generateXml(const std::string& elementName, const T& value){
//	std::ostringstream xml;
//	xml << std::fixed << "<" << elementName << ">" << value << "</" << elementName << ">";
//	return xml.str();
//}



template string generateXml(const std::string& elementName, const unsigned int& value);
template string generateXml(const std::string& elementName, const int& value);
template string generateXml(const std::string& elementName, const double& value);
template string generateXml(const std::string& elementName, const string& value);



template <typename T>
std::string generateXmlVector(const std::string& name, const T& data) {

	// Throw an exception if the container is empty
	if (data.getSize() == 0) {
		throw std::invalid_argument("The data container is empty.");
	}
	std::ostringstream xml;

	xml << "<" << name << ">\n";
	for (unsigned int i=0; i<data.getSize() ; i++) {
		xml << std::fixed << "\t<item>" << data(i) << "</item>\n";
	}

	xml << "</" << name << ">";

	return xml.str();
}


template std::string generateXmlVector(const std::string& name, const vec& data);


template <typename T>
void writeXmlElement(std::ofstream& file, const std::string& elementName, const T& value){
	file << "\t<" << elementName << ">" << value << "</" << elementName << ">" << std::endl;
}

template <typename T>
void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const T& values) {
	file << "\t<" << elementName << ">" << std::endl;
	for (unsigned int i=0; i<values.getSize() ; i++) {
		double val = values(i);
		file << "\t\t<Item>" << val << "</Item>" << std::endl;
	}
	file << "\t</" << elementName << ">" << std::endl;
}


int readIntegerFromXmlFile(ifstream &file, const string keyword) {

	string line;
	string tag;
	string content;

	int output = 0;

	while (std::getline(file, line)) {
		std::istringstream iss(line);
		if (std::getline(iss, tag, '>') && std::getline(iss, content, '<')) {
			tag = removeSpacesFromString(tag);
			content =  removeSpacesFromString(content);

			if (tag == keyword) {
				output = std::stoi(content);
			}
		}
	}

	return output;

}


template void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const vec& values);

template void writeXmlElement(std::ofstream& file, const std::string& elementName, const unsigned int& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const int& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const double& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const string& value);


double getDoubleValueFromXML(const std::string& xmlString, const std::string& keyword) {
	// Find keyword position
	size_t keywordStart = xmlString.find("<" + keyword + ">");
	if (keywordStart == std::string::npos){
		return 10E-14;
	}

	size_t keywordEnd = xmlString.find("</" + keyword + ">", keywordStart);
	if (keywordEnd == std::string::npos) {
		throw std::invalid_argument("Closing tag for keyword not found");
	}

	size_t nextKeywordStart = xmlString.find("<" + keyword + ">", keywordEnd);
	if (nextKeywordStart != std::string::npos) {
		std::string msg = "Multiple occurrences of keyword: " + keyword;
		throw std::invalid_argument(msg);
	}


	// Extract substring containing value
	std::string valueString = xmlString.substr(keywordStart + keyword.length() + 2, keywordEnd - keywordStart - keyword.length() - 2);

	valueString = removeSpacesFromString(valueString);

	double value;
	try
	{
		value = std::stod(valueString);
	}
	catch (...)
	{
		throw std::invalid_argument("Conversion to double has failed.");
	}


	return value;
}







string getStringValueFromXML(const std::string& xmlString, const std::string& keyword) {
	std::string result;

	// Find the starting position of the keyword
	size_t keywordStart = xmlString.find("<" + keyword + ">");
	if (keywordStart == std::string::npos) {
		return result;
	}

	// Find the ending position of the keyword
	size_t keywordEnd = xmlString.find("</" + keyword + ">", keywordStart);
	if (keywordEnd == std::string::npos) {
		throw std::invalid_argument("Closing tag for keyword not found.");
	}

	// Check if there's another occurrence of the keyword
	size_t nextKeywordStart = xmlString.find("<" + keyword + ">", keywordEnd);
	if (nextKeywordStart != std::string::npos) {
		std::string msg = "Multiple occurrences of keyword: " + keyword;
		throw std::invalid_argument(msg);
	}

	// Extract the value between the keyword tags
	size_t valueStart = keywordStart + keyword.length() + 2;
	result = xmlString.substr(valueStart, keywordEnd - valueStart);

	result = removeSpacesFromString(result);

	return result;
}



int getIntegerValueFromXML(const std::string& xmlString, const std::string& keyword) {
	// Find keyword position
	size_t keywordStart = xmlString.find("<" + keyword + ">");
	if (keywordStart == std::string::npos){
		return -10000000;
	}

	size_t keywordEnd = xmlString.find("</" + keyword + ">", keywordStart);
	if (keywordEnd == std::string::npos) {
		throw std::invalid_argument("Closing tag for keyword not found");
	}

	size_t nextKeywordStart = xmlString.find("<" + keyword + ">", keywordEnd);
	if (nextKeywordStart != std::string::npos) {
		throw std::invalid_argument("Multiple occurrences of keyword");
	}


	// Extract substring containing value
	std::string valueString = xmlString.substr(keywordStart + keyword.length() + 2, keywordEnd - keywordStart - keyword.length() - 2);



	int value;


	try
	{
		value = std::stoi(valueString);
	}
	catch (...)
	{
		throw std::invalid_argument("Conversion to integer has failed.");
	}

	return value;
}


std::vector<double> getDoubleVectorValuesFromXML(const std::string& xmlInput, const std::string& keyword) {

	if(xmlInput.empty()){
		throw std::invalid_argument("Xml input is empty.");
	}
	if(keyword.empty()){
		throw std::invalid_argument("Keyword is empty.");
	}

	std::vector<double> result;
	std::string startTag = "<" + keyword + ">";
	std::string endTag = "</" + keyword + ">";
	size_t startPos = xmlInput.find(startTag);
	size_t endPos = xmlInput.find(endTag);

	if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
		std::string content = xmlInput.substr(startPos + startTag.length(), endPos - startPos - startTag.length());

		std::istringstream iss(content);
		std::string item;
		while (std::getline(iss, item, '<')) {
			if (!item.empty()) {
				size_t startPos = item.find('>');
				if (startPos != std::string::npos) {
					std::string number = item.substr(startPos + 1);

					try {
						double value = std::stod(number);
						result.push_back(value);
					} catch (...) {

					}
				}
			}
		}
	}

	return result;
}


std::vector<std::string> getStringVectorValuesFromXML(const std::string& input, const std::string& keyword) {

	if(input.empty()){
		throw std::invalid_argument("Xml input is empty.");
	}
	if(keyword.empty()){
		throw std::invalid_argument("Keyword is empty.");
	}

	std::vector<std::string> contents;
	std::string startTag = "<" + keyword + ">";
	std::string endTag = "</" + keyword + ">";
	size_t startPos = input.find(startTag);
	size_t endPos = input.find(endTag);

	if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
		startPos += startTag.length();
		std::string substr = input.substr(startPos, endPos - startPos);
		size_t pos = 0;
		while ((pos = substr.find("<item>", pos)) != std::string::npos) {
			size_t endItemPos = substr.find("</item>", pos);
			if (endItemPos != std::string::npos) {

				contents.push_back(removeSpacesFromString(substr.substr(pos + 6, endItemPos - pos - 6)));
				pos = endItemPos + 7;
			} else {
				break;
			}
		}
	}
	return contents;
}



} /* Namespace Rodop */

