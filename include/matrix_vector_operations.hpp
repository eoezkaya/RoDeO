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
#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP
#include <armadillo>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <quadmath.h>
using namespace arma;

class HighPrecisionMatrix {
public:
	HighPrecisionMatrix(int rows, int columns){

		Nrows = rows;
		Ncolumns = columns;
		elements = new __float128[rows*columns]();



	}



	int n_rows() const { return Nrows; }
	int n_cols() const { return Ncolumns; }

	const __float128& operator()(int row, int column) const {

		return elements[row * Ncolumns + column];
	}
	__float128& operator()(int row, int column) {

		return elements[row * Ncolumns + column];
	}

	void operator=(mat &A) {

		for(int i=0; i<Nrows; i++){
			for(int j=0; j<Ncolumns; j++){

				elements[i * Ncolumns + j] = A(i,j);
			}
		}


	}

	void print(void){

		std::cout.precision(10);
		for(int i=0; i<Nrows; i++){
			for(int j=0; j<Ncolumns; j++){

				cout<<double(elements[i * Ncolumns + j])<<" ";

			}

			cout<<"\n";
		}

	}

	~HighPrecisionMatrix(){

		delete[] elements;
	}


private:
	int Nrows;
	int Ncolumns;
	__float128 *elements;
};

HighPrecisionMatrix factorizeCholeskyHP(const HighPrecisionMatrix& input);
HighPrecisionMatrix transposeHP(const HighPrecisionMatrix& input);
void testHighPrecisionCholesky(void);

bool isEqual(const mat &A, const mat&B, double tolerance);

void printMatrix(mat M, std::string name="None");
void printVector(vec v, std::string name="None");
void printVector(rowvec v, std::string name="None");
void printVector(std::vector<std::string> v);
void printVector(std::vector<int> v);
void printVector(std::vector<bool> v);

vec normalizeColumnVector(vec x, double xmin, double xmax);

rowvec normalizeRowVector(rowvec x, vec xmin, vec xmax);
rowvec normalizeRowVectorBack(rowvec xnorm, vec xmin, vec xmax);

void copyRowVector(rowvec &a,rowvec b);
void copyRowVector(rowvec &a,rowvec b, unsigned int);

void appendRowVectorToCSVData(rowvec v, std::string fileName);
void saveMatToCVSFile(mat M, std::string fileName);

mat normalizeMatrix(mat matrixIn);
mat normalizeMatrix(mat matrixIn, vec xmin, vec xmax);

#endif
