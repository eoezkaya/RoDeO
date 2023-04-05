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

#ifndef AUX_FUNCTIONS_HPP
#define AUX_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "matrix_vector_operations.hpp"
#include "random_functions.hpp"
#include "metric.hpp"
#include <armadillo>
#include <vector>
#include <map>
#include <math.h>
#include <cassert>
using namespace arma;
using std::string;

void executePythonScript(std::string command);

void compileWithCpp(std::string, std::string);




void changeDirectoryToUnitTests(void);

void normalizeDataMatrix(mat matrixIn, mat &matrixOut);



bool checkValue(double value, double expected, double tolerance);
bool checkValue(double value, double expected);

template<typename T> bool isBetween(T number, T a, T b){

	assert(b>a);

	if(number >= a && number <= b) return true;
	return false;
}


template<typename T> bool isIntheList(const std::vector<T> &vec, T item){

	if ( std::find(vec.begin(), vec.end(), item) != vec.end() )
		return true;
	else
		return false;

}

template<typename T> bool isNotIntheList(const std::vector<T> &vec, T item){

	if(isIntheList(vec, item)) return false;
	else return true;
}


template<typename T> bool isIntheList(T* v, T item, unsigned int dim){

	for(unsigned int i=0; i<dim; i++){

		if(v[i] == item) return true;

	}

	return false;
}

template<typename T> bool isNotIntheList(T* v, T item, unsigned int dim){

	if(isIntheList(v, item, dim)) return false;
	else return true;
}


bool checkMatrix(mat values, mat expected, double tolerance);
bool checkMatrix(mat values, mat expected);

void abortIfDoesNotMatch(int firstNumber, int secondNumber, string message = "None");
void abortWithErrorMessage(string message);


bool isEmpty(std::string);
bool isNotEmpty(std::string);
bool isEqual(string s1, string s2);


bool checkIfOn(std::string keyword);
bool checkIfOff(std::string keyword);



double calculatePolynomial(double x, const rowvec &coeffs);
double calculateTensorProduct(const rowvec &x, const mat &coeffs);

double pdf(double x, double mu, double sigma);

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma);
double calculateProbalityLessThanAValue(double value, double mu, double sigma);
double calculateProbalityGreaterThanAValue(double value, double mu, double sigma);




//void solveLinearSystemCholesky(mat U, vec &x, vec b);

bool file_exist(const char *fileName);
bool file_exist(std::string filename);

void readFileToaString(std::string, std::string &);


std::vector<std::string> getStringValuesFromString(std::string sub_str, char delimiter);
vec getDoubleValuesFromString(std::string sub_str, char delimiter);
std::string removeSpacesFromString(std::string );
std::string removeKeywordFromString(std::string inputStr,  std::string keyword);

void compute_max_min_distance_data(mat &x, double &max_distance, double &min_distance);


bool checkifTooCLose(const rowvec &, const rowvec &, double = 10E-6);
bool checkifTooCLose(const rowvec &, const mat &,double = 10E-6);


#define PRINT_HERE printf("Here : %s(%d)--:\n",__FILE__,__LINE__);


#endif
