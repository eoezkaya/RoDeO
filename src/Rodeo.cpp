/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
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



#include<stdio.h>
#include<iostream>
#include "auxiliary_functions.hpp"
#include "drivers.hpp"
#include "output.hpp"
#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif



int main(int argc, char* argv[]){


	printRoDeOIntro();
	/* initialize random seed*/
	srand (time(NULL));

#ifdef UNIT_TESTS

	changeDirectoryToUnitTests();

	testing::InitGoogleTest(&argc, argv);
	int runTestsResult = RUN_ALL_TESTS();

	return runTestsResult;

#endif


	if(argc == 1){
		abortWithErrorMessage("File name for the configuration file (*.cfg) is missing!");
	}

	std::string fileNameConfig = argv[1];


	RoDeODriver driverToRun;
	driverToRun.setConfigFilename(fileNameConfig);
	driverToRun.readConfigFile();

	driverToRun.run();

	return 0;

}
