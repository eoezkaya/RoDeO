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

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "auxiliary_functions.hpp"
#include "drivers.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>



RoDeODriver::RoDeODriver(){

	dimension = 0;
	numberOfKeywords = 16;
	problemName = "None";
	problemType = "None";
	numberOfConstraints = 0;
	objectiveFunctionName = "Objective function";
	ifObjectiveFunctionNameIsSet = false;
	ifObjectiveFunctionOutputFileIsSet = false;
	ifNumberOfConstraintsSet = false;
	ifProblemDimensionSet = false;
	ifUpperBoundsSet = false;
	ifLowerBoundsSet = false;
	ifProblemTypeSet = false;

	IsGradientsAvailableForObjectiveFunction = false;

	keywords = new std::string[numberOfKeywords];
	keywords[0]="PROBLEM_TYPE=";
	keywords[1]="DIMENSION=";
	keywords[2]="NUMBER_OF_CONSTRAINTS=";
	keywords[3]="UPPER_BOUNDS=";
	keywords[4]="LOWER_BOUNDS=";
	keywords[5]="OBJECTIVE_FUNCTION_EXECUTABLE_NAME=";
	keywords[6]="CONSTRAINT_EXECUTABLE_NAMES=";
	keywords[7]="CONSTRAINT_DEFINITIONS=";
	keywords[8]="PROBLEM_NAME=";
	keywords[9]="OBJECTIVE_FUNCTION_NAME=";
	keywords[10]="OBJECTIVE_FUNCTION_EXECUTABLE_PATH=";
	keywords[11]="CONSTRAINT_EXECUTABLE_PATHS=";
	keywords[12]="OBJECTIVE_FUNCTION_OUTPUT_FILENAME=";
	keywords[13]="CONSTRAIN_FUNCTION_OUTPUT_FILENAMES=";
	keywords[14]="OBJECTIVE_FUNCTION_GRADIENT=";
	keywords[15]="CONSTRAINT_GRADIENTS=";

	configFileName = settings.config_file;

}

void RoDeODriver::readConfigFile(void){

	if(!file_exist(configFileName)){

		std::cout<<"ERROR: Configuration file does not exist!\n";
		abort();
	}

#if 1
	printf("reading configuration file...\n");
#endif
	size_t len = 0;
	ssize_t readlen;
	char * line = NULL;
	FILE *inp = fopen(configFileName.c_str(),"r");

	if (inp == NULL){

		std::cout<<"ERROR: Configuration file cannot be opened!\n";
		abort();
	}

	while ((readlen = getline(&line, &len, inp)) != -1) {


		if(line[0]!= '#'){
			std::string str(line);

#if 0
			printf("Retrieved line of length %zu :\n", readlen);
			printf("%s\n", str.c_str());
#endif

			for(unsigned int key=0; key<numberOfKeywords; key++){
#if 0
				printf("searching the keyword: %s\n", keywords[key].c_str());
#endif
				std::size_t found = str.find(keywords[key]);

				if (found!=std::string::npos){
#if 0
					printf("found the keyword: %s\n", keywords[key].c_str());
					printf("found = %d\n",found);
#endif
					str.erase(std::remove_if(str.begin(), str.end(), isspace), str.end());
					std::string sub_str = str.substr(found+keywords[key].length());
#if 0
					printf("keyword:%s\n", sub_str.c_str());
#endif
					switch(key){
					case 0: {
						problemType = sub_str;
						std::cout<<"PROBLEM_TYPE= "<<problemType<<"\n";
						ifProblemTypeSet = true;
						break;
					}

					case 1: {
						dimension = std::stoi(sub_str);
						std::cout<<"DIMENSION= "<<dimension<<"\n";
						ifProblemDimensionSet = true;
						break;
					}
					case 2: {
						numberOfConstraints = std::stoi(sub_str);
						ifNumberOfConstraintsSet = true;
						break;
					}
					case 3: {
						sub_str.erase(std::remove_if(sub_str.begin(), sub_str.end(), ::isspace), sub_str.end());
#if 0
						std::cout<<sub_str;
						std::cout << sub_str.back() << '\n';
#endif

						if(sub_str.back() != '}'){

							bool doneFlag=true;

							while(doneFlag){


								getline(&line, &len, inp);
								std::string str(line);
								str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
								sub_str += str;
#if 0
								std::cout<<"str = "<<str<<"a\n";
								std::cout<<"str.back() = "<< str.back() << '\n';
#endif
								if(str.back() == '}') doneFlag = false;



							}




						}


						std::vector<std::string> valuesReadFromString;

						getValuesFromString(sub_str,valuesReadFromString,',');


						boxConstraintsUpperBounds = zeros<vec>(valuesReadFromString.size());

						for (int i = 0; i<valuesReadFromString.size() ; i++){

							boxConstraintsUpperBounds(i) = std::stoi(valuesReadFromString[i]);

						}
						std::cout<<"UPPER_BOUNDS=";
						trans(boxConstraintsUpperBounds).print();
						ifUpperBoundsSet = true;

						break;
					}/* end of case 3 */

					case 4: {
						sub_str.erase(std::remove_if(sub_str.begin(), sub_str.end(), ::isspace), sub_str.end());

						if(sub_str.back() != '}'){

							bool doneFlag=true;

							while(doneFlag){


								getline(&line, &len, inp);
								std::string str(line);
								str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
								sub_str += str;

								if(str.back() == '}') doneFlag = false;

							}


						}

						std::vector<std::string> valuesReadFromString;

						getValuesFromString(sub_str,valuesReadFromString,',');


						boxConstraintsLowerBounds = zeros<vec>(valuesReadFromString.size());

						for (int i = 0; i<valuesReadFromString.size() ; i++){

							boxConstraintsLowerBounds(i) = std::stoi(valuesReadFromString[i]);

						}
						std::cout<<"LOWER_BOUNDS=";
						trans(boxConstraintsLowerBounds).print();
						ifLowerBoundsSet = true;

						break;
					} /* end of case 4 */

					case 5: {
						executableNames.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_EXECUTABLE_NAME= "<<executableNames.front()<<"\n";
						ifObjectiveFunctionNameIsSet = true;
						break;
					}

					case 6: {
						std::cout<<"here\n";
						std::cout<<sub_str<<"\n";
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');

						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executableNames.push_back(*it);

						}


						std::cout<<"CONSTRAINT_EXECUTABLE_NAMES=";
						for (auto it = executableNames.begin()+1; it != executableNames.end(); it++){

							std::cout<<*it<<" ";

						}
						std::cout<<"\n";

						break;
					}

					case 7: {
						std::cout<<"here\n";
						std::cout<<sub_str<<"\n";
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						int numberOfConstraintsRead = valuesReadFromString.size();

						constraintValues = zeros<vec>(numberOfConstraintsRead);
						int count = 0;
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							std::string s = *it;
							std::cout<<s<<"\n";
							std::size_t found  = s.find(">");
							std::size_t found2 = s.find("<");
							if (found!=std::string::npos){
								std::cout << "> found at: " << found << '\n';
								std::string name,type, value;
								name.assign(s,0,found);
								type.assign(s,found,1);
								value.assign(s,found+1,s.length() - name.length() - type.length());

								std::cout<<"name = "<<name<<"\n";
								std::cout<<"type = "<<type<<"\n";
								std::cout<<"value = "<<value<<"\n";
								constraintNames.push_back(name);
								constraintTypes.push_back(type);
								constraintValues(count) = std::stod(value);
								count++;
							}
							else if (found2!=std::string::npos){
								std::cout << "< found at: " << found2 << '\n';
								std::string name,type, value;
								name.assign(s,0,found2);
								type.assign(s,found2,1);
								value.assign(s,found2+1,s.length() - name.length() - type.length());

								std::cout<<"name = "<<name<<"\n";
								std::cout<<"type = "<<type<<"\n";
								std::cout<<"value = "<<value<<"\n";
								constraintNames.push_back(name);
								constraintTypes.push_back(type);
								constraintValues(count) = std::stod(value);
								count++;
							}
							else{

								std::cout<<"ERROR: Format error in CONSTRAINT_EXECUTABLE_NAMES!\n";
								abort();

							}



						}



						break;
					}

					case 8: {
						problemName = sub_str;
						std::cout<<"PROBLEM_NAME= "<<problemName<<"\n";
						break;
					}


					case 9: {
						objectiveFunctionName = sub_str;
						std::cout<<"OBJECTIVE_FUNCTION_NAME= "<<objectiveFunctionName<<"\n";

						break;
					}

					case 10: {
						executablePaths.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_EXECUTABLE_PATH= "<<executablePaths.front()<<"\n";
						break;
					}

					case 11: {
						std::cout<<"here\n";
						std::cout<<sub_str<<"\n";
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						std::cout<<"CONSTRAINT_FUNCTION_EXECUTABLE_PATHS= ";
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executablePaths.push_back(*it);
							std::cout<<*it<<" ";

						}
						std::cout<<"\n";


						break;
					}

					case 12: {
						executableOutputFiles.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_OUTPUT_FILENAME= "<<executableOutputFiles.front()<<"\n";
						ifObjectiveFunctionOutputFileIsSet = true;
						break;
					}

					case 13: {
						std::cout<<"here\n";
						std::cout<<sub_str<<"\n";
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						std::cout<<"CONSTRAINT_FUNCTION_OUTPUT_FILENAMES= ";
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executableOutputFiles.push_back(*it);
							std::cout<<*it<<" ";

						}
						std::cout<<"\n";


						break;
					}

					case 14: {
						if(sub_str == "yes" ||  sub_str == "YES" || sub_str == "y" || sub_str == "Y" || sub_str == "Yes" ){

							IsGradientsAvailableForObjectiveFunction = true;
							std::cout<<"OBJECTIVE_FUNCTION_GRADIENT= "<<IsGradientsAvailableForObjectiveFunction<<"\n";

						}


						else if(sub_str == "no" ||  sub_str == "NO" || sub_str == "n" ||  sub_str == "N"|| sub_str == "No"){
							std::cout<<"OBJECTIVE_FUNCTION_GRADIENT= "<<IsGradientsAvailableForObjectiveFunction<<"\n";
						}
						else{

							std::cout<<"ERROR: Unknown keyword for OBJECTIVE_FUNCTION_GRADIENT\n";
							abort();
						}

						break;
					}



					}

				}
			}

		} /* end of if */

	} /* end of while */

	fclose(inp);

	checkConsistencyOfConfigParams();





}

void RoDeODriver::checkConsistencyOfConfigParams(void){

	int numberOfExeNames = executableNames.size();
	int numberOfExeOutputFile = executableOutputFiles.size();


	if(!ifProblemTypeSet){

		std::cout<<"ERROR: Problem type is defined, did you set PROBLEM_TYPE?\n";
		abort();
	}

	if(!ifLowerBoundsSet){

		std::cout<<"ERROR: Lower bounds for the optimization parameters are not set, did you set LOWER_BOUNDS properly?\n";
		abort();

	}

	if(!ifUpperBoundsSet){

		std::cout<<"ERROR: Upper bounds for the optimization parameters are not set, did you set UPPER_BOUNDS properly?\n";
		abort();

	}

	if(boxConstraintsLowerBounds.size() != dimension){

		std::cout<<"ERROR: Number of lower bounds does not match with the problem dimension, did you set LOWER_BOUNDS properly?\n";
		abort();

	}
	if(boxConstraintsUpperBounds.size() != dimension){

		std::cout<<"ERROR: Number of upper bounds does not match with the problem dimension, did you set UPPER_BOUNDS properly?\n";
		abort();

	}

	/* check if box constraints are set properly */

	if(ifLowerBoundsSet && ifUpperBoundsSet){

		for(unsigned int i=0; i<dimension; i++){

			if(boxConstraintsLowerBounds(i) >= boxConstraintsUpperBounds(i)){

				std::cout<<"ERROR: Lower bounds cannot be greater or equal than upper bounds, did you set LOWER_BOUNDS and UPPER_BOUNDS properly?\n";
				abort();


			}


		}



	}




	if(!ifProblemDimensionSet){

		std::cout<<"ERROR: Number of optimization parameters is not set, did you set DIMENSION properly?\n";
		abort();


	}

	if(!ifObjectiveFunctionNameIsSet){

		std::cout<<"ERROR: Objective function is required, did you set OBJECTIVE_FUNCTION_EXECUTABLE_NAME properly?\n";
		abort();
	}


	if(!ifObjectiveFunctionOutputFileIsSet){

		std::cout<<"ERROR: Objective function output file is required, did you set OBJECTIVE_FUNCTION_OUTPUT_FILENAME properly?\n";
		abort();
	}

	if(!ifNumberOfConstraintsSet){

		std::cout<<"ERROR: number of constraints is not set, did you set NUMBER_OF_CONSTRAINTS properly?\n";
		abort();

	}


	if(numberOfExeNames != numberOfConstraints+1){

		std::cout<<"ERROR: Number of executable names does not match with the number of constraints, did you set CONSTRAINT_EXECUTABLE_NAMES properly?\n";
		abort();

	}






}



void RoDeODriver::runDriver(void){




}


