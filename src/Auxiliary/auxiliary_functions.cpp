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

#include "./INCLUDE/auxiliary_functions.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"
#include "../INCLUDE/Rodeo_globals.hpp"
#include <string>
#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
#include <stdio.h>


using std::string;

#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#define CHANGE_DIR _chdir
#define popen _popen
#else
#include <unistd.h>
#define CHANGE_DIR chdir
#endif

bool changeDirectory(const std::string& directory) {
    if (CHANGE_DIR(directory.c_str()) != 0) {
        std::cerr << "Error: Could not change directory to " << directory << std::endl;
        return false;
    }
    return true;
}

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



void changeDirectoryToUnitTests(void){

	int ret =chdir ("./UnitTests");
	if(ret!=0){

		std::cout<<"ERROR: Cannot change directory to $./UnitTests\n";
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








