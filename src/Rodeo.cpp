
/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */



#include<stdio.h>
#include<iostream>


#include "optimization.hpp"
#include "test_functions.hpp"
#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "auxiliary_functions.hpp"
#include "read_settings.hpp"
#include "surrogate_model.hpp"

#include "aggregation_model.hpp"
#include "kriging_training.hpp"
#include "polynomials.hpp"
#include "polynomials_test.hpp"
#include "gek_test.hpp"
#include "lhs.hpp"
#include "drivers.hpp"
#include<gtest/gtest.h>
Rodeo_settings settings;




int main(int argc, char* argv[]){


	printf("\n\n\n");


	printf("	  ____       ____        ___   \n");
	printf("	 |  _ \\ ___ |  _ \\  ___ / _ \\  \n");
	printf("	 | |_) / _ \\| | | |/ _ \\ | | | \n");
	printf("	 |  _ < (_) | |_| |  __/ |_| | \n");
	printf("	 |_| \\_\\___/|____/ \\___|\\___/  \n");

	printf("\n");
	printf("    RObust DEsign Optimization Package      ");
	printf("\n\n\n");




	/* initialize random seed*/
	srand (time(NULL));

	settings.read();



#ifdef UNIT_TESTS


	chdir ("../UnitTests");
	testing::InitGoogleTest(&argc, argv);
	int runTestsResult = RUN_ALL_TESTS();


	return runTestsResult;

#endif


	int ret = chdir (settings.cwd.c_str());

	if (ret != 0){

		cout<<"Error: Cannot change directory! Are you sure that the directory: "<<settings.cwd<<" exists?\n";
		abort();
	}





	RoDeODriver driverToRun;
	driverToRun.readConfigFile();

	driverToRun.runDriver();




}




