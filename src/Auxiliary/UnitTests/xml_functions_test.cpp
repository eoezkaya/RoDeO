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

#include "xml_functions.hpp"

#include<gtest/gtest.h>

using namespace std;
#include <armadillo>
using namespace arma;

TEST(testXMLFunctions, writeXmlElement){


	std::ofstream file("test.xml");
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << "test.xml" << std::endl;
		return;
	}

	file << "<Design>" << std::endl;

	unsigned int ID = 5;
	writeXmlElement(file, "DesignID", ID);

	double number = 7.88;
	writeXmlElement(file, "Value", number);

	string str = "test";

	writeXmlElement(file, "Name", str);

	file << "</Design>" << std::endl;
	file.close();


}

TEST(testXMLFunctions, writeXmlElementVector){


	std::ofstream file("test.xml");
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << "test.xml" << std::endl;
		return;
	}

	file << "<Design>" << std::endl;

	rowvec v(3); v(0) = 0.01; v(1) = -2; v(2) = 77.2121;
	writeXmlElementVector(file, "vector", v);
	vec w(3); w(0) = 0.01; w(1) = -2; w(2) = 77.2121;
	writeXmlElementVector(file, "vector", w);
	file << "</Design>" << std::endl;
	file.close();


}

TEST(testXMLFunctions, generateXml){
	std::string value = "emre";
	std::string name = "name";
	std::string text = generateXml(name,value);
//	std::cout<<text<<"\n";
	text = generateXml(name,4);
//	std::cout<<text<<"\n";
	text = generateXml(name,4.55510E7);
	std::cout<<text<<"\n";

	rowvec a(3);
	a(0) = 0.11;
	a(1) = 0.17;
	a(2) = -1.11;

	text = generateXmlVector(name,a);
	std::cout<<text<<"\n";


}



