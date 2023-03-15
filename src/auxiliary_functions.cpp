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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "auxiliary_functions.hpp"
#include <chrono>
#include <random>
#include <string>
#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include<algorithm>
#include<cctype>


using std::string;



void executePythonScript(std::string command){

	FILE* in = popen(command.c_str(), "r");
	fprintf(in, "\n");

}

void compileWithCpp(std::string fileName, std::string exeName){

	assert(!fileName.empty());
	assert(!exeName.empty());

	if(!file_exist(fileName)){

		std::cout<<"ERROR: File "<<fileName<<" does not exist\n";
		abort();

	}

	std::string compileCommand = "g++ "+ fileName + " -o " + exeName + " -lm";
	system(compileCommand.c_str());



}

void changeDirectoryToRodeoHome(void){

	const char* env_p;
	if(env_p = std::getenv("RODEO_HOME")){
		std::cout << "RODEO_HOME: " << env_p << '\n';
	}
	else{
		std::cout<<"The environmental variable RODEO_HOME is undefined!\n";
		abort();

	}

	int ret = chdir (env_p);
	if(ret!=0){

		std::cout<<"ERROR: Cannot change directory to $RODEO_HOME\n";
		abort();
	}

}

void changeDirectoryToUnitTests(void){

	int ret =chdir ("./UnitTests");
	if(ret!=0){

		std::cout<<"ERROR: Cannot change directory to $RODEO_HOME/UnitTests\n";
		abort();
	}

}

void changeDirectoryToWork(std::string cwd){

	int ret = chdir (cwd.c_str());

	if (ret != 0){

		cout<<"Error: Cannot change directory! Are you sure that the directory: "<<cwd<<" exists?\n";
		abort();
	}

}

bool checkValue(double value, double expected, double tolerance){

	assert(tolerance > 0.0);

	if(fabs(value-expected) > tolerance) {
#if 0
		printf("\nvalue = %10.7f, expected = %10.7f, error = %10.7f, tolerance = %10.7f\n",value, expected,fabs(value-expected),tolerance );
#endif
		return false;
	}
	else return true;


}





bool isEqual(string s1, string s2){

	if(s1.length()!= s2.length()) return false;

	if(s1.compare(s2) == 0 ) return true;
	else return false;

}


bool checkValue(double value, double expected){

	double tolerance = 0.0;
	if(fabs(value) < 10E-10){

		tolerance = EPSILON;

	}
	else{

		tolerance = fabs(value) * 0.001;
	}

	double error = fabs(value-expected);
	if(error > tolerance) {

#if 0
		printf("\nvalue = %10.7f, expected = %10.7f, error = %10.7f, tolerance = %10.7f\n",value, expected,error, tolerance);
#endif
		return false;
	}
	else return true;


}

bool checkMatrix(mat values, mat expected, double tolerance){
	assert(values.n_rows == expected.n_rows);
	assert(values.n_cols == expected.n_cols);
	bool result = true;

	for(unsigned int i=0; i<values.n_rows; i++){

		for(unsigned int j=0; j<values.n_cols; j++){

			if(!checkValue(values(i,j), expected(i,j), tolerance)){

				result = false;
			}

		}

	}

	return result;

}


bool checkMatrix(mat values, mat expected){
	assert(values.n_rows == expected.n_rows);
	assert(values.n_cols == expected.n_cols);
	bool result = true;

	for(unsigned int i=0; i<values.n_rows; i++){

		for(unsigned int j=0; j<values.n_cols; j++){

			if(!checkValue(values(i,j), expected(i,j))){

				result = false;
			}

		}

	}

	return result;

}


void abortIfDoesNotMatch(int firstNumber, int secondNumber, string message ){

	if(firstNumber != secondNumber){

		std::cout<<"ERROR: "<<message<<"\n";
		abort();
	}


}

void abortWithErrorMessage(string message){

	std::cout<<"ERROR: "<<message<<"\n";
	abort();


}



bool isEmpty(std::string inputStr){

	if(inputStr.empty()){

		return true;
	}
	else{

		return false;
	}

}


bool isNotEmpty(std::string inputStr){

	if(inputStr.empty()){

		return false;
	}
	else{

		return true;
	}

}


bool checkIfOn(std::string keyword){

	bool flag = false;

	if(keyword == "YES") flag = true;
	if(keyword == "Yes") flag = true;
	if(keyword == "yes") flag = true;
	if(keyword == "ON") flag = true;
	if(keyword == "On") flag = true;
	if(keyword == "on") flag = true;
	if(keyword == "y") flag = true;
	if(keyword == "Y") flag = true;

	return flag;

}

bool checkIfOff(std::string keyword){

	bool flag = false;

	if(keyword == "NO") flag = true;
	if(keyword == "No") flag = true;
	if(keyword == "no") flag = true;
	if(keyword == "Off") flag = true;
	if(keyword == "OFF") flag = true;
	if(keyword == "off") flag = true;
	if(keyword == "N") flag = true;
	if(keyword == "n") flag = true;

	return flag;

}





/** Returns the pdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_pdf(x) with mu and sigma
 *
 */


double pdf(double x, double m, double s)
{
	double a = (x - m) / s;

	return INVSQRT2PI / s * std::exp(-0.5 * a * a);
}



/** Returns the cdf of x, given the distribution described by mu and sigma..
 *
 *  CFD(x0) = Pr(x < x0)
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_cdf(x) with mu and sigma
 *
 */
double cdf(double x0, double mu, double sigma)
{

	double inp = (x0 - mu) / (sigma * SQRT2);
	double result = 0.5 * (1.0 + erf(inp));
	return result;
}



double calculateProbalityLessThanAValue(double value, double mu, double sigma){

	double p = cdf(value, mu, sigma);
	return p;


}
double calculateProbalityGreaterThanAValue(double value, double mu, double sigma){

	double p =  1.0 - cdf(value, mu, sigma);
	return p;

}




bool checkifTooCLose(const rowvec &v1, const rowvec &v2, double tol){

	rowvec diff = v1 - v2;

	double distance = norm(diff, 1);

	if(distance < tol) return true;
	else return false;

}

bool checkifTooCLose(const rowvec &v1, const mat &M, double tol){


	unsigned int nRows = M.n_rows;
	bool ifTooClose = false;

	for(unsigned int i=0; i<nRows; i++){

		rowvec r = M.row(i);
		ifTooClose = checkifTooCLose(v1,r, tol);

		if(ifTooClose) {
			break;
		}

	}

	return ifTooClose;
}




bool file_exist(std::string filename)
{

	return file_exist(filename.c_str());
}

bool file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}



void readFileToaString(std::string filename, std::string & stringCompleteFile){

	if(!file_exist(filename)){

		std::cout<<"ERROR: File "<< filename <<" does not exist!\n";
		abort();
	}

	std::ifstream configfile(filename);

	if(configfile) {
		std::ostringstream ss;
		ss << configfile.rdbuf();
		stringCompleteFile = ss.str();
	}

#if 0
	cout<<stringCompleteFile;
#endif


}



std::string removeSpacesFromString(std::string inputString){

	inputString.erase(std::remove_if(inputString.begin(), inputString.end(), ::isspace), inputString.end());
	return inputString;
}



std::string removeKeywordFromString(std::string inputStr,  std::string keyword){

	assert(!keyword.empty());
	assert(!inputStr.empty());

	std::size_t found = inputStr.find(keyword);

	if(found != std::string::npos){

		inputStr.erase(std::remove_if(inputStr.begin(), inputStr.end(), ::isspace), inputStr.end());
		std::string sub_str = inputStr.substr(found+keyword.length() + 1);


		return sub_str;


	}
	else{

		return inputStr;
	}



}


std::vector<std::string> getStringValuesFromString(std::string str,char delimiter){

	str = removeSpacesFromString(str);

	std::vector<std::string> values;


	if(str[0] == '{' || str[0] == '['){

		str.erase(0,1);
	}

	if(str[str.length()-1] == '}' || str[str.length()-1] == ']'){

		str.erase(str.length()-1,1);
	}

	while(1){

		std::size_t found = str.find(delimiter);
		if (found==std::string::npos) break;

		std::string buffer;
		buffer.assign(str,0,found);
		str.erase(0,found+1);


		values.push_back(buffer);

	}

	values.push_back(str);

	return values;
}



vec getDoubleValuesFromString(std::string str,char delimiter){

	str = removeSpacesFromString(str);

	size_t n = std::count(str.begin(), str.end(), ',');

	vec values(n+1);


	if(str[0] == '{' || str[0] == '['){

		str.erase(0,1);
	}

	if(str[str.length()-1] == '}' || str[str.length()-1] == ']'){

		str.erase(str.length()-1,1);
	}

	int count = 0;
	while(1){

		std::size_t found = str.find(delimiter);
		if (found==std::string::npos) break;

		std::string buffer;
		buffer.assign(str,0,found);
		str.erase(0,found+1);


		values(count) = std::stod(buffer);
		count ++;
	}
	values(count) = std::stod(str);

	return values;
}





