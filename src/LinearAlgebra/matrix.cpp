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

#include "./INCLUDE/matrix.hpp"
//#include "./INCLUDE/vector.hpp"
#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For seeding srand()
#include <cassert>
using namespace rodeo;

void mat::allocateMemory() {

	assert(nrows>0);
	assert(ncols>0);

	matrix = new double*[nrows];
	for (int i = 0; i < nrows; ++i) {
		matrix[i] = new double[ncols];
	}
}

// Constructor

mat::mat(int defaultRows, int defaultCols) : nrows(defaultRows), ncols(defaultCols), matrix(nullptr) {
	if (nrows > 0 && ncols > 0) {
		allocateMemory();
		fill(0.0);
	}
}

mat::mat(int r, int c, double val) : nrows(r), ncols(c) {
	allocateMemory();
	fill(val);
}


// Copy constructor
mat::mat(const mat& other) : nrows(other.nrows), ncols(other.ncols) {

	// Allocate memory for the new matrix
	matrix = new double*[nrows];
	for (int i = 0; i < nrows; ++i) {
		matrix[i] = new double[ncols];

		// Copy the elements from the other matrix
		for (int j = 0; j < ncols; ++j) {
			matrix[i][j] = other.matrix[i][j];
		}
	}
}

// Destructor
mat::~mat() {

	if(matrix!=nullptr){

		for (int i = 0; i < nrows; ++i) {
			delete[] matrix[i];
		}
		delete[] matrix;

	}
}

int mat::getNRows(void) const{
	return nrows;
}
int mat::getNCols(void) const{
	return ncols;
}

int mat::getSize(void) const{

	return nrows*ncols;
}

// Matrix addition
mat mat::operator+(const mat& other) const {
	mat result(nrows, ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
		}
	}
	return result;
}

// Matrix subtraction
mat mat::operator-(const mat& other) const {
	mat result(nrows, ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result.matrix[i][j] = matrix[i][j] - other.matrix[i][j];
		}
	}
	return result;
}

// Matrix multiplication
mat mat::operator*(const mat& other) const {
	mat result(nrows, other.ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < other.ncols; ++j) {
			result.matrix[i][j] = 0;
			for (int k = 0; k < ncols; ++k) {
				result.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
			}
		}
	}
	return result;
}

mat mat::operator*(const double scalar) const {
	mat result(nrows, ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result.matrix[i][j] = matrix[i][j] * scalar;
		}
	}
	return result;
}


// Display matrix
void mat::print() const {
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

// Access operator
double& mat::operator()(int i, int j) {
	return matrix[i][j];
}

// Const version of the access operator for read-only access
const double& mat::operator()(int i, int j) const {
	return matrix[i][j];

}

void mat::fill(double value) {
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			matrix[i][j] = value;
		}
	}
}

// Fill method to randomly fill the entries
void mat::fillRandom(void) {

	assert(ncols>0);
	assert(nrows>0);
	// Seed the random number generator
	std::srand(std::time(0));

	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			// Generate a random double between 0.0 and 1.0
			matrix[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
		}
	}
}

// Add this member function implementation to the mat.cpp file

// Resize method to change the number of rows and columns
void mat::resize(int newRows, int newCols) {
	// Create a new matrix with the specified size
	double** newMatrix = new double*[newRows];
	for (int i = 0; i < newRows; ++i) {
		newMatrix[i] = new double[newCols];
	}
	for (int i = 0; i < newRows; ++i)
		for (int j = 0; j < newCols; ++j) newMatrix[i][j] = 0.0;


	// Copy the elements from the existing matrix to the new matrix
	int minRows = std::min(nrows, newRows);
	int minCols = std::min(ncols, newCols);

	for (int i = 0; i < minRows; ++i) {
		for (int j = 0; j < minCols; ++j) {
			newMatrix[i][j] = matrix[i][j];
		}
	}

	// Deallocate memory for the existing matrix
	deallocateMemory();

	// Update the mat object with the new matrix and size
	matrix = newMatrix;
	nrows = newRows;
	ncols = newCols;
}


// Subview method to create a view of a portion of the matrix
mat mat::submat(int startRow, int startCol, int subRows, int subCols) const {

	assert(startRow + subRows <= nrows);
	assert(startCol + subCols <= ncols);

	mat subMatrix(subRows, subCols);

	for (int i = 0; i < subRows; ++i) {
		for (int j = 0; j < subCols; ++j) {
			subMatrix(i, j) = matrix[startRow + i][startCol + j];
		}
	}

	return subMatrix;
}

void mat::deallocateMemory(void) {
	// Deallocate memory for the existing matrix
	for (int i = 0; i < nrows; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

// Add columns method to add columns to the matrix
void mat::addColumns(int numColumnsToAdd, double defaultValue) {
	// Create a new matrix with the updated size
	double** newMatrix = new double*[nrows];
	for (int i = 0; i < nrows; ++i) {
		newMatrix[i] = new double[ncols + numColumnsToAdd];
	}

	// Copy the elements from the existing matrix to the new matrix
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			newMatrix[i][j] = matrix[i][j];
		}

		// Set the new columns to the specified default value
		for (int k = ncols; k < ncols + numColumnsToAdd; ++k) {
			newMatrix[i][k] = defaultValue;
		}
	}

	// Deallocate memory for the existing matrix
	deallocateMemory();
	// Update the mat object with the new matrix and size
	matrix = newMatrix;
	ncols += numColumnsToAdd;
}

// Add rows method to add rows to the matrix
void mat::addRows(int numRowsToAdd, double defaultValue) {
	// Create a new matrix with the updated size
	double** newMatrix = new double*[nrows + numRowsToAdd];
	for (int i = 0; i < nrows + numRowsToAdd; ++i) {
		newMatrix[i] = new double[ncols];
	}

	// Copy the elements from the existing matrix to the new matrix
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			newMatrix[i][j] = matrix[i][j];
		}
	}

	// Set the new rows to the specified default value
	for (int i = nrows; i < nrows + numRowsToAdd; ++i) {
		for (int j = 0; j < ncols; ++j) {
			newMatrix[i][j] = defaultValue;
		}
	}

	deallocateMemory();
	// Update the mat object with the new matrix and size
	matrix = newMatrix;
	nrows += numRowsToAdd;
}

// Add this member function implementation to the mat.cpp file

// Concatenate method to concatenate two matrices row-wise
mat mat::concatenateRowWise(const mat& other) const {

	assert(ncols>0);
	assert(nrows>0);
	assert(ncols == other.ncols);

	mat result(nrows + other.nrows, ncols);

	// Copy the elements from the first matrix to the result matrix
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result(i, j) = matrix[i][j];
		}
	}

	// Copy the elements from the second matrix to the result matrix
	for (int i = 0; i < other.nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result(i + nrows, j) = other(i, j);
		}
	}

	return result;
}


// Concatenate method to concatenate two matrices column-wise
mat mat::concatenateColumnWise(const mat& other) const {

	assert(ncols>0);
	assert(nrows>0);
	assert(nrows == other.nrows);

	mat result(nrows, ncols + other.ncols);

	// Copy the elements from the first matrix to the result matrix
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			result(i, j) = matrix[i][j];
		}
	}

	// Copy the elements from the second matrix to the result matrix
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < other.ncols; ++j) {
			result(i, j + ncols) = other(i, j);
		}
	}

	return result;
}

//vec mat::getRow(int n) const {
//
//	assert(n>0);
//	assert(n<nrows);
//
//    // Create a new vector for the row
//    rodeo::vec rowVector(ncols);
//
//    // Copy elements from the matrix row to the vector
//    for (int i = 0; i < ncols; ++i) {
//        rowVector(i) = matrix[n][i];
//    }
//
//    return rowVector;
//}


