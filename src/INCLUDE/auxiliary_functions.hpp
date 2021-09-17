/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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



void changeDirectoryToRodeoHome(void);
void changeDirectoryToUnitTests(void);
void changeDirectoryToWork(std::string cwd);

void perturbVectorUniform(frowvec &xp,float sigmaPert);

void normalizeDataMatrix(mat matrixIn, mat &matrixOut);



bool checkValue(double value, double expected, double tolerance);
bool checkValue(double value, double expected);
bool checkMatrix(mat values, mat expected, double tolerance);
bool checkMatrix(mat values, mat expected);

void abortIfDoesNotMatch(int firstNumber, int secondNumber, string message = "None");

bool isEmpty(std::string);
bool isNotEmpty(std::string);

bool checkIfOn(std::string keyword);
bool checkIfOff(std::string keyword);



double calculatePolynomial(double x, const rowvec &coeffs);
double calculateTensorProduct(const rowvec &x, const mat &coeffs);

double pdf(double x, double mu, double sigma);

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma);





void solveLinearSystemCholesky(mat U, vec &x, vec b);

bool file_exist(const char *fileName);
bool file_exist(std::string filename);

void readFileToaString(std::string, std::string &);


std::vector<std::string> getStringValuesFromString(std::string sub_str, char delimiter);
vec getDoubleValuesFromString(std::string sub_str, char delimiter);
std::string removeSpacesFromString(std::string );
std::string removeKeywordFromString(std::string inputStr,  std::string keyword);


int check_if_lists_are_equal(int *list1, int *list2, int dim);
int is_in_the_list(int entry, int *list, int list_size);
int is_in_the_list(int entry, std::vector<int> &list);
int is_in_the_list(unsigned int entry, uvec &list);

bool ifIsInTheList(const std::vector<std::string> &vec, std::string item);


void compute_max_min_distance_data(mat &x, double &max_distance, double &min_distance);


void generate_validation_set(int *indices, int size, int N);
void generate_validation_set(uvec &indices, int size);

void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod);
void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod, uvec &map);


bool checkifTooCLose(const rowvec &, const rowvec &, double = 10E-6);
bool checkifTooCLose(const rowvec &, const mat &,double = 10E-6);


bool checkLinearSystem(mat A, vec x, vec b, double tol);
vec calculateResidual(mat A, vec x, vec b);



/* distance functions */

template<class T>
double L1norm(T x, int p, int* index=NULL){
	double sum=0.0;
	if(index == NULL){

		for(int i=0;i<p;i++){

			sum+=fabs(x(i));
		}
	}
	else{

		for(int i=0;i<p;i++){

			sum+=fabs(x(index[i]));
		}

	}

	return sum;
}


template<class T>
double L2norm(T x, int p, int* index=NULL){

	double sum;
	if(index == NULL){
		sum=0.0;

		for(int i=0;i<p;i++){

			sum+=x(i)*x(i);
		}

	}
	else{

		sum=0.0;

		for(int i=0;i<p;i++){

			sum+=x(index[i])*x(index[i]);
		}

	}

	return sqrt(sum);
}

template<class T>
double Lpnorm(T x, int p, int size,int *index=NULL){
	double sum=0.0;


	if(index == NULL){

		for(int i=0;i<size;i++){

			sum+=pow(fabs(x(i)),p);
		}

	}
	else{

		for(int i=0;i<size;i++){

			sum+=pow(fabs(x(index[i])),p);
		}


	}
	return pow(sum,1.0/p);
}


void findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices, unsigned int norm=2);


void findKNeighbours(mat &data,
		             rowvec &p,
					 int K,
					 vec &min_dist,
					 uvec &indices,
					 mat M);


void findKNeighbours(mat &data,
		rowvec &p,
		int K,
		int *input_indx ,
		double* min_dist,
		int *indices,
		int number_of_independent_variables);

int getPopularlabel(int* labels, int size);

void testLPnorm(void);

double compute_R(rowvec x_i, rowvec x_j, vec theta, vec gamma);
void compute_R_matrix(vec theta, vec gamma, double reg_param,mat& R, mat &X);

//void compute_R_matrix_GEK(vec theta, double reg_param, mat& R, mat &X, mat &grad);


double compute_R_Gauss(rowvec x_i, rowvec x_j, vec theta);
double compR_dxi(rowvec x_i, rowvec x_j, vec theta, int k);

#endif
