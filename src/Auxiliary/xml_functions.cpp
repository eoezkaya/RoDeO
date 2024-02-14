/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "./INCLUDE/xml_functions.hpp"
#include "./INCLUDE/auxiliary_functions.hpp"

using namespace std;

#include <armadillo>
using namespace arma;



template <typename T>
string generateXml(const std::string& elementName, const T& value){
	std::ostringstream xml;
	    xml << std::fixed << "<" << elementName << ">" << value << "</" << elementName << ">";
	    return xml.str();
}



template string generateXml(const std::string& elementName, const unsigned int& value);
template string generateXml(const std::string& elementName, const int& value);
template string generateXml(const std::string& elementName, const double& value);
template string generateXml(const std::string& elementName, const string& value);



template <typename T>
std::string generateXmlVector(const std::string& name, const T& data) {

	assert(data.size() > 0);
    std::ostringstream xml;

    xml << "<" << name << ">\n";
    for (unsigned int i=0; i<data.size(); i++) {
        xml << std::fixed << "\t<item>" << data(i) << "</item>\n";
    }

    xml << "</" << name << ">";

    return xml.str();
}


template std::string generateXmlVector(const std::string& name, const rowvec& data);
template std::string generateXmlVector(const std::string& name, const vec& data);


template <typename T>
void writeXmlElement(std::ofstream& file, const std::string& elementName, const T& value){
	file << "\t<" << elementName << ">" << value << "</" << elementName << ">" << std::endl;
}

template <typename T>
void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const T& values) {
	file << "\t<" << elementName << ">" << std::endl;
	for (unsigned int i=0; i<values.size(); i++) {
		double val = values(i);
		file << "\t\t<Item>" << val << "</Item>" << std::endl;
	}
	file << "\t</" << elementName << ">" << std::endl;
}


int readIntegerFromXmlFile(ifstream &file, const string keyword) {

	string line;
	string tag;
	string content;

	int output;

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




template void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const rowvec& values);
template void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const vec& values);

template void writeXmlElement(std::ofstream& file, const std::string& elementName, const unsigned int& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const int& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const double& value);
template void writeXmlElement(std::ofstream& file, const std::string& elementName, const string& value);
