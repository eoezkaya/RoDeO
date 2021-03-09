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
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "auxiliary_functions.hpp"
#include "drivers.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>



RoDeODriver::RoDeODriver(){

	dimension = 0;
	numberOfKeywords = 23;
	problemName = "None";
	problemType = "None";
	designVectorFilename = "None";
	numberOfConstraints = 0;
	objectiveFunctionName = "Objective function";
	ifObjectiveFunctionNameIsSet = false;
	ifObjectiveFunctionOutputFileIsSet = false;
	ifDesignVectorFileNameSet = false;
	ifNumberOfConstraintsSet = false;
	ifProblemDimensionSet = false;
	ifUpperBoundsSet = false;
	ifLowerBoundsSet = false;
	ifProblemTypeSet = false;
	ifmaximumNumberOfSimulationsSet = false;
	ifmaximumNumberOfDoESamplesSet = false;
	ifexecutablePathObjectiveFunctionSet = false;
	ifSurrogateModelTypeSet = false;

	ifWarmStart = false;


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
	keywords[13]="CONSTRAINT_FUNCTION_OUTPUT_FILENAMES=";
	keywords[16]="MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS=";
	keywords[17]="NUMBER_OF_DOE_SAMPLES=";
	keywords[18]="WARM_START=";
	keywords[19]="DESIGN_VECTOR_FILENAME=";
	keywords[20]="GRADIENT_AVAILABLE=";
	keywords[21]="CLEAN_DOE_FILES=";
	keywords[22]="SURROGATE_MODEL=";


	configFileName = settings.config_file;

}

void RoDeODriver::readConfigFile(void){

	bool ifParameterAlreadySet[numberOfKeywords];

	for(unsigned int i=0; i<numberOfKeywords; i++){

		ifParameterAlreadySet[i] = false;
	}


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
						if(checkifProblemTypeIsValid(problemType)){

							std::cout<<"PROBLEM_TYPE= "<<problemType<<"\n";
							ifProblemTypeSet = true;
						}
						else{

							std::cout<<"ERROR: Unknown keyword for PROBLEM_TYPE\n";
							std::cout<<"Valid keywords: DoE, MINIMIZATION, MAXIMIZATION\n";
							abort();

						}

						break;
					}

					case 1: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: DIMENSION is already defined in the config file!\n";
							abort();
						}

						dimension = std::stoi(sub_str);
						std::cout<<"DIMENSION= "<<dimension<<"\n";
						ifProblemDimensionSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}
					case 2: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: NUMBER_OF_CONSTRAINTS is already defined in the config file!\n";
							abort();
						}

						numberOfConstraints = std::stoi(sub_str);
						std::cout<<"NUMBER_OF_CONSTRAINTS= "<<numberOfConstraints<<"\n";
						ifNumberOfConstraintsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}
					case 3: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: UPPER_BOUNDS are already defined in the config file!\n";
							abort();
						}

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

						for (unsigned int i = 0; i<valuesReadFromString.size() ; i++){

							boxConstraintsUpperBounds(i) = std::stod(valuesReadFromString[i]);

						}
						std::cout<<"UPPER_BOUNDS=";
						trans(boxConstraintsUpperBounds).print();
						ifUpperBoundsSet = true;
						ifParameterAlreadySet[key] = true;

						break;
					}/* end of case 3 */

					case 4: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: LOWER_BOUNDS are already defined in the config file!\n";
							abort();
						}

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

						for (unsigned int i = 0; i<valuesReadFromString.size() ; i++){

							boxConstraintsLowerBounds(i) = std::stod(valuesReadFromString[i]);

						}
						std::cout<<"LOWER_BOUNDS=";
						trans(boxConstraintsLowerBounds).print();
						ifLowerBoundsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					} /* end of case 4 */

					case 5: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"OBJECTIVE_FUNCTION_EXECUTABLE_NAME is already defined in the config file!\n";
							abort();
						}
						executableNames.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_EXECUTABLE_NAME= "<<executableNames.front()<<"\n";
						ifObjectiveFunctionNameIsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 6: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"CONSTRAINT_EXECUTABLE_NAMES are already defined in the config file!\n";
							abort();
						}
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
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 7: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: CONSTRAINT_EXECUTABLE_NAMES are already defined in the config file!\n";
							abort();
						}
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						int numberOfConstraintsRead = valuesReadFromString.size();

						constraintValues = zeros<vec>(numberOfConstraintsRead);
						int count = 0;
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							std::string s = *it;

							std::size_t found  = s.find(">");
							std::size_t found2 = s.find("<");
							if (found!=std::string::npos){

								std::string name,type, value;
								name.assign(s,0,found);
								type.assign(s,found,1);
								value.assign(s,found+1,s.length() - name.length() - type.length());
#if 0
								std::cout<<"name = "<<name<<"\n";
								std::cout<<"type = "<<type<<"\n";
								std::cout<<"value = "<<value<<"\n";
#endif
								constraintNames.push_back(name);
								constraintTypes.push_back(type);
								constraintValues(count) = std::stod(value);
								count++;
							}
							else if (found2!=std::string::npos){

								std::string name,type, value;
								name.assign(s,0,found2);
								type.assign(s,found2,1);
								value.assign(s,found2+1,s.length() - name.length() - type.length());
#if 0
								std::cout<<"name = "<<name<<"\n";
								std::cout<<"type = "<<type<<"\n";
								std::cout<<"value = "<<value<<"\n";
#endif
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


						ifParameterAlreadySet[key] = true;
						break;
					}

					case 8: {

						problemName = sub_str;
						std::cout<<"PROBLEM_NAME= "<<problemName<<"\n";
						ifParameterAlreadySet[key] = true;
						break;
					}


					case 9: {
						objectiveFunctionName = sub_str;
						std::cout<<"OBJECTIVE_FUNCTION_NAME= "<<objectiveFunctionName<<"\n";
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 10: {
						executablePaths.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_EXECUTABLE_PATH= "<<executablePaths.front()<<"\n";
						ifexecutablePathObjectiveFunctionSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 11: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: CONSTRAINT_FUNCTION_EXECUTABLE_PATHS are already defined in the config file!\n";
							abort();
						}
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						std::cout<<"CONSTRAINT_FUNCTION_EXECUTABLE_PATHS= ";
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executablePaths.push_back(*it);
							std::cout<<*it<<" ";

						}
						std::cout<<"\n";

						ifParameterAlreadySet[key] = true;
						break;
					}

					case 12: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: OBJECTIVE_FUNCTION_OUTPUT_FILENAME is already defined in the config file!\n";
							abort();
						}
						executableOutputFiles.push_back(sub_str);
						std::cout<<"OBJECTIVE_FUNCTION_OUTPUT_FILENAME= "<<executableOutputFiles.front()<<"\n";
						ifObjectiveFunctionOutputFileIsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 13: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: CONSTRAINT_FUNCTION_OUTPUT_FILENAMES are already defined in the config file!\n";
							abort();
						}
						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						std::cout<<"CONSTRAINT_FUNCTION_OUTPUT_FILENAMES= ";
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executableOutputFiles.push_back(*it);
							std::cout<<*it<<" ";

						}
						std::cout<<"\n";
						ifConstraintFunctionOutputFileIsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 14: {

						ifParameterAlreadySet[key] = true;
						break;
					}

					case 16: {

						maximumNumberOfSimulations = std::stoi(sub_str);
						std::cout<<"MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS= "<<maximumNumberOfSimulations<<"\n";
						ifmaximumNumberOfSimulationsSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 17: {
						maximumNumberDoESamples = std::stoi(sub_str);
						std::cout<<"NUMBER_OF_DOE_SAMPLES= "<<maximumNumberDoESamples<<"\n";
						ifmaximumNumberOfDoESamplesSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 18: {
						if(sub_str == "YES" || sub_str == "yes") {
							this->ifWarmStart = true;
						}
						else if(sub_str == "NO" || sub_str == "no"){
							this->ifWarmStart = false;
						}
						else{
							std::cout<<"ERROR: Unknown keyword for WARM_START\n";
							abort();

						}
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 19: {
						designVectorFilename = sub_str;
						std::cout<<"DESIGN_VECTOR_FILENAME= "<<designVectorFilename<<"\n";
						ifDesignVectorFileNameSet = true;
						ifParameterAlreadySet[key] = true;
						break;
					}

					case 20: {
						if(ifParameterAlreadySet[key]){

							std::cout<<"ERROR: GRADIENT_AVAILABLE are already defined in the config file!\n";
							abort();
						}

						std::vector<std::string> valuesReadFromString;
						getValuesFromString(sub_str,valuesReadFromString,',');
						std::cout<<"GRADIENT_AVAILABLE= ";
						for (auto it = valuesReadFromString.begin(); it != valuesReadFromString.end(); it++){

							executablesWithGradient.push_back(*it);
							std::cout<<*it<<" ";

						}
						std::cout<<"\n";

						ifParameterAlreadySet[key] = true;
						break;
					}

					case 21: {
						if(sub_str == "YES" || sub_str == "yes") {
							ifDoEFilesShouldBeCleaned = true;
						}
						else if(sub_str == "NO" || sub_str == "no"){

							ifDoEFilesShouldBeCleaned = false;
						}
						else{

							std::cout<<"Not valid input for CLEAN_DOE_FILES\n";
							abort();
						}
						break;
					}
					case 22: {

						surrogateModelType = sub_str;
						std::cout<<"SURROGATE_MODEL= "<<surrogateModelType<<"\n";
						ifSurrogateModelTypeSet = true;
						break;
					}



					}

				}
			}

		} /* end of if */

	} /* end of while */

	fclose(inp);


	std::vector<std::string>::iterator it = find (executablesWithGradient.begin(), executablesWithGradient.end(), objectiveFunctionName);

	if (it != executablesWithGradient.end()){

		IsGradientsAvailable.push_back(true);

	}
	else{

		IsGradientsAvailable.push_back(false);
	}




	for(unsigned int i=0; i<constraintNames.size(); i++){

		std::string constraintName = constraintNames.at(i);

		it = find (executablesWithGradient.begin(), executablesWithGradient.end(), constraintName);

		if (it != executablesWithGradient.end()){

			IsGradientsAvailable.push_back(true);

		}
		else{

			IsGradientsAvailable.push_back(false);
		}





	}
	printVector(IsGradientsAvailable);



	checkConsistencyOfConfigParams();





}

void RoDeODriver::checkConsistencyOfConfigParams(void) const{

	unsigned int numberOfExeNames = executableNames.size();


	if(!ifProblemTypeSet){

		std::cout<<"ERROR: Problem type is defined, did you set PROBLEM_TYPE?\n";
		abort();
	}

	if(problemType != "DoE" && !ifmaximumNumberOfSimulationsSet){

		std::cout<<"ERROR: Computational budget is not set, did you set MAXIMUM_NUMBER_OF_FUNCTION_EVALUATIONS?\n";
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

	if(!ifDesignVectorFileNameSet){

		std::cout<<"ERROR: Design vector file name is required, did you set DESIGN_VECTOR_FILENAME properly?\n";
		abort();
	}




	if(numberOfExeNames != numberOfConstraints+1){

		std::cout<<"ERROR: Number of executable names does not match with the number of constraints, did you set CONSTRAINT_EXECUTABLE_NAMES properly?\n";
		abort();

	}

	if(problemType != "DoE" && !ifWarmStart && !ifmaximumNumberOfDoESamplesSet){

		std::cout<<"ERROR: Without warm start number of samples for the DoE must be specified, did you set NUMBER_OF_DOE_SAMPLES properly?\n";
		abort();

	}


	if(problemType == "DoE" && !ifmaximumNumberOfDoESamplesSet){

		std::cout<<"ERROR: The number of samples for the DoE must be specified, did you set NUMBER_OF_DOE_SAMPLES properly?\n";
		abort();
	}


	if(numberOfConstraints != constraintValues.size()){

		std::cout<<"ERROR: There is some problem with constraint definitions, did you set CONSTRAINT_DEFINITIONS properly?\n";
		abort();

	}

	if(numberOfConstraints > 0 && !ifConstraintFunctionOutputFileIsSet){

		std::cout<<"ERROR: Output files for the constraints are not set, did you set CONSTRAINT_FUNCTION_OUTPUT_FILENAMES properly?\n";
		abort();

	}

	if( (numberOfConstraints+1) != executableOutputFiles.size()){

		std::cout<<"ERROR: There is some problem with output file definitions, did you set OBJECTIVE_FUNCTION_OUTPUT_FILENAME and CONSTRAINT_FUNCTION_OUTPUT_FILENAMES properly?\n";
		abort();

	}


}


bool RoDeODriver::checkifProblemTypeIsValid(std::string s) const{

	if (s == "DoE" || s == "MINIMIZATION" || s == "MAXIMIZATION"){

		return true;
	}
	else return false;


}

void RoDeODriver::setObjectiveFunction(ObjectiveFunction & objFunc){

	objFunc.setExecutableName(executableNames.front());
	if(ifexecutablePathObjectiveFunctionSet){

		objFunc.setExecutablePath(executablePaths.front());

	}

	objFunc.setFileNameReadInput(executableOutputFiles.front());
	objFunc.setFileNameDesignVector(designVectorFilename);
	objFunc.setParameterBounds(boxConstraintsLowerBounds,boxConstraintsUpperBounds);

	/* switch on gradients if they are available */


#if 0
	printVector(executablesWithGradient);
#endif
	if ( std::find(executablesWithGradient.begin(), executablesWithGradient.end(), objectiveFunctionName) != executablesWithGradient.end() ){

		objFunc.setGradientOn();
	}

	objFunc.print();

}





void RoDeODriver::setConstraint(ConstraintFunction & constraintFunc, unsigned int indx){


	std::string exeName = executableNames.at(indx);
	std::string outputFileName = executableOutputFiles.at(indx);
	constraintFunc.setExecutableName(exeName);

	constraintFunc.setFileNameReadInput(executableOutputFiles.at(indx));

	constraintFunc.setFileNameDesignVector(designVectorFilename);
	constraintFunc.setID(indx);
	constraintFunc.setParameterBounds(boxConstraintsLowerBounds,boxConstraintsUpperBounds);



	/* check whether executable is already specified for another constraint or objective function */


	for(unsigned int i=0; i<executableNames.size(); i++){

		if (i!= indx){

			if(exeName == executableNames.at(i)){

				std::cout<<"same exe names: "<<executableNames.at(i)<<" "<<exeName<<"\n";
				constraintFunc.IDToFunctionsShareOutputExecutable.push_back(i);

			}

		}

	}

	unsigned int indxStartToRead = 0;
	for(unsigned int i=0; i<executableOutputFiles.size(); i++){

		if (i!= indx){

			if(outputFileName == executableOutputFiles.at(i)){


				if(i < indx){
					if(IsGradientsAvailable.at(i)){

						indxStartToRead+= dimension+1;


					}
					else{

						indxStartToRead++;

					}
				}


			}

		}





	}


	constraintFunc.readOutputStartIndex = indxStartToRead;

	if ( std::find(executablesWithGradient.begin(), executablesWithGradient.end(), constraintFunc.getName()) != executablesWithGradient.end() ){

#if 1
		std::cout<<"Gradients available for the constraint: "<<constraintFunc.getName()<<"\n";
#endif
		constraintFunc.setGradientOn();


	}





}


void RoDeODriver::runOptimization(void) const{

	//	optimizationStudy.setProblemType(problemType);
	//	optimizationStudy.ifVisualize = true;
	//	optimizationStudy.setMaximumNumberOfIterations(maximumNumberOfSimulations);
	//
	//	if(!ifWarmStart){
	//
	//		optimizationStudy.performDoE(maximumNumberDoESamples,LHS);
	//
	//	}
	//
	//
	//	if(!optimizationStudy.checkSettings()){
	//		std::cout<<"ERROR: Check of the settings is failed!\n";
	//		abort();
	//	}
	//	optimizationStudy.EfficientGlobalOptimization();



}

void RoDeODriver::runSurrogateModelTest(void) const{

	TestFunction TestFunction(objectiveFunctionName, dimension);



}



void RoDeODriver::runDriver(void){

	COptimizer optimizationStudy(problemName, dimension, problemType);
	optimizationStudy.setBoxConstraints(boxConstraintsLowerBounds,boxConstraintsUpperBounds);
	optimizationStudy.setFileNameDesignVector(designVectorFilename);


	ObjectiveFunction objFunc(objectiveFunctionName, dimension);
	setObjectiveFunction(objFunc);
	optimizationStudy.addObjectFunction(objFunc);




	for(unsigned int i=0; i<numberOfConstraints; i++){

		std::string name = constraintNames.at(i);
		std::string constraintType = constraintTypes.at(i);
		double constraintValue = constraintValues(i);
#if 0
		std::cout<<name<<" "<<constraintType<<" "<<constraintValue<<"\n";
#endif
		ConstraintFunction constraintFunc(name,dimension);

		std::string inequalityStatement = constraintType+std::to_string(constraintValue);
#if 0
		std::cout<<inequalityStatement<<"\n";
#endif
		constraintFunc.setInequalityConstraint(inequalityStatement);

		setConstraint(constraintFunc,i+1);
		optimizationStudy.addConstraint(constraintFunc);


	}

#if 1
	optimizationStudy.printConstraints();
#endif

	if(ifDoEFilesShouldBeCleaned){

		optimizationStudy.ifCleanDoeFiles = true;
	}else{

		optimizationStudy.ifCleanDoeFiles = false;
	}


	if(problemType == "DoE"){


		if(!optimizationStudy.checkSettings()){
			std::cout<<"ERROR: Check of the settings is failed!\n";
			abort();
		}
		optimizationStudy.performDoE(maximumNumberDoESamples,LHS);


	}
	else if(problemType == "MAXIMIZATION" || problemType == "MINIMIZATION"){

		optimizationStudy.setProblemType(problemType);
		optimizationStudy.ifVisualize = true;
		optimizationStudy.setMaximumNumberOfIterations(maximumNumberOfSimulations);

		if(!ifWarmStart){

			optimizationStudy.performDoE(maximumNumberDoESamples,LHS);

		}


		if(!optimizationStudy.checkSettings()){
			std::cout<<"ERROR: Check of the settings is failed!\n";
			abort();
		}
		optimizationStudy.EfficientGlobalOptimization();


	}
	else if(problemType == "SURROGATE_TEST"){

		runSurrogateModelTest();




	}

	else{

		std::cout<<"ERROR: PROBLEM_TYPE is unknown\n";
		abort();

	}
}


