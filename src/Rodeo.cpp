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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include "./Driver/INCLUDE/driver_xml.hpp"


bool isXMLFile(const std::string& fileName) {
    // Check if the file name ends with ".xml" ignoring case
    return fileName.length() >= 4 &&
           std::equal(fileName.end() - 4, fileName.end(), ".xml", [](char a, char b) {
               return std::tolower(a) == std::tolower(b);
           });
}

int main(int argc, char* argv[]) {
    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Check if configuration file is provided
    if (argc == 1) {
        std::cerr << "Error: File name for the configuration file (*.xml) is missing!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string fileNameConfig = argv[1];

    try {
        if (isXMLFile(fileNameConfig)) {
            // Initialize the XML driver
            Rodop::Driver driverXML;
            driverXML.setConfigFileName(fileNameConfig);
            driverXML.readConfigurationFile();
            driverXML.run();
        } else {
            throw std::runtime_error("Wrong format for the configuration file!");
        }
    } catch (const std::exception& e) {
        // Catch and report errors
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Indicate success
    return EXIT_SUCCESS;
}

