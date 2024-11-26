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
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm> // For std::swap
#include <random>    // For std::random_device and std::mt19937
#include <vector>
#include <fstream>

using namespace std;

#ifdef OPENBLAS

#include <cblas.h>
extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b, const int* ldb, int* info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
}
#endif

namespace Rodop {

// Constructor

mat::mat(unsigned int rows, unsigned int cols) : nrows(rows), ncols(cols) {
	matrix = new double[rows * cols];
	fill(0.0);
}

mat::~mat() {
	delete[] matrix;
}


mat::mat()
: nrows(0), ncols(0), matrix(nullptr) {}



mat::mat(unsigned int rows, unsigned int cols, double val)
: nrows(rows), ncols(cols) {
	matrix = new double[rows * cols];
	fill(val);
}

mat::mat(const mat& other) : nrows(other.nrows), ncols(other.ncols) {
	matrix = new double[nrows * ncols];
	std::copy(other.matrix, other.matrix + nrows * ncols, matrix);
}


double* mat::getPointer() const{
	return matrix;
}

void mat::reset() {
	// Free the allocated memory if it exists
	if (matrix != nullptr) {
		delete[] matrix;
		matrix = nullptr;
	}

	// Reset the dimensions to zero
	nrows = 0;
	ncols = 0;
}


mat& mat::operator=(const mat& other) {
	if (this == &other) {
		return *this; // Handle self-assignment
	}

	delete[] matrix; // Free the existing resource

	nrows = other.nrows;
	ncols = other.ncols;
	matrix = new double[nrows * ncols];
	std::copy(other.matrix, other.matrix + nrows * ncols, matrix);

	return *this;
}

mat& mat::operator=(mat&& other) noexcept {
	if (this != &other) {
		delete[] matrix; // Free the existing resource

		nrows = other.nrows;
		ncols = other.ncols;
		matrix = other.matrix;

		other.nrows = 0;
		other.ncols = 0;
		other.matrix = nullptr;
	}
	return *this;
}



mat::mat(mat&& other) noexcept : nrows(other.nrows), ncols(other.ncols), matrix(other.matrix) {
	other.nrows = 0;
	other.ncols = 0;
	other.matrix = nullptr;
}




double& mat::operator()(unsigned int row, unsigned int col) {
	if (row >= nrows || col >= ncols) {
		throw std::out_of_range("Index out of range.");
	}
	return matrix[col * nrows + row];
}

const double& mat::operator()(unsigned int row, unsigned int col) const {
	if (row >= nrows || col >= ncols) {
		throw std::out_of_range("Index out of range.");
	}
	return matrix[col * nrows + row];
}

mat mat::submat(unsigned int startRow, unsigned int endRow, unsigned int startCol, unsigned int endCol) const {
	// Validate indices
	if (startRow >= nrows ||
			endRow >= nrows ||
			startCol >= ncols ||
			endCol >= ncols ||
			endRow < startRow ||
			endCol < startCol)
	{
		throw std::invalid_argument("Invalid submatrix dimensions or indices: startRow=" + std::to_string(startRow) +
				", endRow=" + std::to_string(endRow) + ", startCol=" + std::to_string(startCol) +
				", endCol=" + std::to_string(endCol) + ".");
	}

	unsigned int numRows = endRow - startRow + 1;
	unsigned int numCols = endCol - startCol + 1;

	mat subMatrix(numRows, numCols);

	for (unsigned int j = 0; j < numCols; ++j) {
		for (unsigned int i = 0; i < numRows; ++i) {
			subMatrix(i, j) = matrix[(startCol + j) * nrows + (startRow + i)];
		}
	}

	return subMatrix;
}


void mat::addRow(const vec& rowVec, int position) {

	if(position == -1){

		position = nrows;
	}

	if(isEmpty()){
		ncols = rowVec.getSize();
	}

	if (position > static_cast<int>(nrows)) {
		throw std::invalid_argument("Position out of range.");
	}
	if (rowVec.getSize() != ncols) {
		throw std::invalid_argument("Row vector size does not match the number of columns.");
	}

	// Create a new matrix with one additional row
	double* newMatrix = new double[(nrows + 1) * ncols];
	if (newMatrix == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Copy the data from the old matrix to the new one, inserting the new row
	for (unsigned int col = 0; col < ncols; ++col) {
		for (int row = 0; row < position; ++row) {
			newMatrix[col * (nrows + 1) + row] = matrix[col * nrows + row];
		}
		// Insert the new row
		newMatrix[col * (nrows + 1) + position] = rowVec(col);
		for (unsigned int row = position; row < nrows; ++row) {
			newMatrix[col * (nrows + 1) + row + 1] = matrix[col * nrows + row];
		}
	}

	// Free the old matrix and update the matrix object
	delete[] matrix;
	matrix = newMatrix;
	++nrows;
}


void mat::addColumn(const vec& colVec, int position) {

	if(position == -1){

		position = ncols;
	}

	if(isEmpty()){
		nrows = colVec.getSize();
	}


	if (position > static_cast<int>(ncols)) {
		throw std::invalid_argument("Position out of range.");
	}
	if (colVec.getSize() != nrows) {
		throw std::invalid_argument("Column vector size does not match the number of rows.");
	}

	// Create a new matrix with one additional column
	double* newMatrix = new double[nrows * (ncols + 1)];
	if (newMatrix == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Copy the data from the old matrix to the new one, inserting the new column
	for (int col = 0; col < position; ++col) {
		for (unsigned int row = 0; row < nrows; ++row) {
			newMatrix[col * nrows + row] = matrix[col * nrows + row];
		}
	}
	// Insert the new column
	for (unsigned int row = 0; row < nrows; ++row) {
		newMatrix[position * nrows + row] = colVec(row);
	}
	for (unsigned int col = position; col < ncols; ++col) {
		for (unsigned int row = 0; row < nrows; ++row) {
			newMatrix[(col + 1) * nrows + row] = matrix[col * nrows + row];
		}
	}

	// Free the old matrix and update the matrix object
	delete[] matrix;
	matrix = newMatrix;
	++ncols;
}


void mat::deleteRow(int rowIndex) {
	/* This is for easy usage */
	if(rowIndex == -1){
		rowIndex = nrows-1;
	}

	if (rowIndex >= static_cast<int>(nrows)) {
		throw std::out_of_range("Row index out of range.");
	}


	if(nrows - 1 > 0){
		// Create a new matrix with one less row
		double* newMatrix = new double[(nrows - 1) * ncols];
		if (newMatrix == nullptr) {
			throw std::runtime_error("Memory allocation failed.");
		}

		// Copy the data from the old matrix to the new one, skipping the specified row
		for (unsigned int col = 0; col < ncols; ++col) {
			for (int row = 0; row < rowIndex; ++row) {
				newMatrix[col * (nrows - 1) + row] = matrix[col * nrows + row];
			}
			for (unsigned int row = rowIndex + 1; row < nrows; ++row) {
				newMatrix[col * (nrows - 1) + (row - 1)] = matrix[col * nrows + row];
			}
		}

		// Free the old matrix and update the matrix object
		delete[] matrix;
		matrix = newMatrix;
		--nrows;

	}
	else{

		delete[] matrix;
		matrix = nullptr;
		nrows = 0;
		ncols = 0;
	}
}

void mat::deleteRows(const std::vector<int>& rowIndices) {
	// Check for empty input
	if (rowIndices.empty()) {
		return; // Nothing to delete
	}

	// Create a sorted list of unique indices to delete (to avoid duplications and to optimize performance)
	std::vector<int> sortedIndices = rowIndices;
	std::sort(sortedIndices.begin(), sortedIndices.end());
	auto last = std::unique(sortedIndices.begin(), sortedIndices.end());
	sortedIndices.erase(last, sortedIndices.end());

	// Check if any index is out of range
	if (sortedIndices.back() >= static_cast<int>(nrows) || sortedIndices.front() < 0) {
		throw std::out_of_range("One or more row indices are out of range.");
	}

	unsigned int newRowCount = nrows - sortedIndices.size();
	if (newRowCount == 0) {
		// If all rows are deleted, reset the matrix
		delete[] matrix;
		matrix = nullptr;
		nrows = 0;
		ncols = 0;
		return;
	}

	// Create a new matrix with the remaining rows
	double* newMatrix = new double[newRowCount * ncols];
	if (newMatrix == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Copy the rows that are not being deleted
	unsigned int newRow = 0;
	unsigned int deleteIndex = 0;
	for (unsigned int oldRow = 0; oldRow < nrows; ++oldRow) {
		if (deleteIndex < sortedIndices.size() && static_cast<int>(oldRow) == sortedIndices[deleteIndex]) {
			++deleteIndex; // Skip this row
		} else {
			for (unsigned int col = 0; col < ncols; ++col) {
				newMatrix[col * newRowCount + newRow] = matrix[col * nrows + oldRow];
			}
			++newRow;
		}
	}

	// Free the old matrix and update the matrix object
	delete[] matrix;
	matrix = newMatrix;
	nrows = newRowCount;
}



void mat::resize(unsigned int newRows, unsigned int newCols) {
	// Create a new matrix with the specified dimensions
	double* newMatrix = new double[newRows * newCols];
	if (newMatrix == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Initialize the new matrix elements to zero
	std::fill(newMatrix, newMatrix + newRows * newCols, 0.0);

	// Copy the data from the old matrix to the new one
	unsigned int minRows = std::min(nrows, newRows);
	unsigned int minCols = std::min(ncols, newCols);

	for (unsigned int col = 0; col < minCols; ++col) {
		for (unsigned int row = 0; row < minRows; ++row) {
			newMatrix[col * newRows + row] = matrix[col * nrows + row];
		}
	}

	// Free the old matrix and update the matrix object
	delete[] matrix;
	matrix = newMatrix;
	nrows = newRows;
	ncols = newCols;
}


void mat::fill(double value) {
	std::fill(matrix, matrix + nrows * ncols, value);
}

void mat::fillRandom() {
	for (unsigned int i = 0; i < nrows*ncols; ++i) {
		matrix[i] = static_cast<double>(rand()) / RAND_MAX;
	}
}

void mat::fillRandom(double a, double b) {

	if (a>=b) {
		throw std::invalid_argument("mat::fillRandom: Upper bound must be larger than lower bound.");
	}


	for (unsigned int i = 0; i < nrows*ncols; ++i) {
		matrix[i] = static_cast<double>(rand()) / RAND_MAX;
		matrix[i] = matrix[i]*(b-a) + a;
	}



}

void mat::fillRandom(const vec& lb, const vec& ub) {
	if (lb.getSize() != ncols || ub.getSize() != ncols) {
		throw std::invalid_argument("Input vectors must have the same size as the number of columns in the matrix.");
	}

	for (unsigned int col = 0; col < ncols; ++col) {
		if (lb(col) > ub(col)) {
			throw std::invalid_argument("For each column, the lower bound must be less than or equal to the upper bound.");
		}

		for (unsigned int row = 0; row < nrows; ++row) {
			double randomValue = static_cast<double>(rand()) / RAND_MAX;
			(*this)(row, col) = lb(col) + randomValue * (ub(col) - lb(col));
		}
	}
}

void mat::fillRandom(const std::vector<double>& lb, const std::vector<double>& ub) {
	if (lb.size() != ncols || ub.size() != ncols) {
		throw std::invalid_argument("Input vectors must have the same size as the number of columns in the matrix.");
	}

	// Random number generator
	std::random_device rd; // Seed for the random number generator
	std::mt19937 gen(rd()); // Mersenne Twister random number generator

	for (unsigned int col = 0; col < ncols; ++col) {
		if (lb[col] > ub[col]) {
			throw std::invalid_argument("For each column, the lower bound must be less than or equal to the upper bound.");
		}

		std::uniform_real_distribution<double> dis(lb[col], ub[col]);

		for (unsigned int row = 0; row < nrows; ++row) {
			(*this)(row, col) = dis(gen); // Generate a random number within [lb[col], ub[col]]
		}
	}
}


void mat::fillRandomLHS(const vec& lb, const vec& ub) {
	if (lb.getSize() != ncols || ub.getSize() != ncols) {
		throw std::invalid_argument("Input vectors must have the same size as the number of columns in the matrix.");
	}

	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Iterate over each column
	for (unsigned int col = 0; col < ncols; ++col) {
		if (lb(col) > ub(col)) {
			throw std::invalid_argument("For each column, the lower bound must be less than or equal to the upper bound.");
		}

		// Create the intervals and randomize them
		std::vector<double> intervals(nrows);
		double intervalSize = (ub(col) - lb(col)) / nrows;

		for (unsigned int row = 0; row < nrows; ++row) {
			double lower = lb(col) + row * intervalSize;
			double upper = lb(col) + (row + 1) * intervalSize;
			std::uniform_real_distribution<double> dis(lower, upper);
			intervals[row] = dis(gen);  // Sample randomly within each interval
		}

		// Shuffle the interval samples
		std::shuffle(intervals.begin(), intervals.end(), gen);

		// Assign the shuffled samples to the matrix
		for (unsigned int row = 0; row < nrows; ++row) {
			(*this)(row, col) = intervals[row];
		}
	}
}


void mat::fillRandomLHS(const std::vector<double>& lb, const std::vector<double>& ub) {
	if (lb.size() != ncols || ub.size() != ncols) {
		throw std::invalid_argument("Input vectors must have the same size as the number of columns in the matrix.");
	}

	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Iterate over each column
	for (unsigned int col = 0; col < ncols; ++col) {
		if (lb[col] > ub[col]) {
			throw std::invalid_argument("For each column, the lower bound must be less than or equal to the upper bound.");
		}

		// Create the intervals and randomize them
		std::vector<double> intervals(nrows);
		double intervalSize = (ub[col] - lb[col]) / nrows;

		for (unsigned int row = 0; row < nrows; ++row) {
			double lower = lb[col] + row * intervalSize;
			double upper = lb[col] + (row + 1) * intervalSize;
			std::uniform_real_distribution<double> dis(lower, upper);
			intervals[row] = dis(gen);  // Sample randomly within each interval
		}

		// Shuffle the interval samples
		std::shuffle(intervals.begin(), intervals.end(), gen);

		// Assign the shuffled samples to the matrix
		for (unsigned int row = 0; row < nrows; ++row) {
			(*this)(row, col) = intervals[row];
		}
	}
}

void mat::eye(){

	if (nrows  != ncols ) {
		throw std::invalid_argument("mat::eye: Matrix must be square.");
	}
	for (unsigned int row = 0; row < nrows; ++row) {
		matrix[row * nrows + row] = 1.0;

	}

}


vec mat::getCol(int colIndex) const {
	if (colIndex >= static_cast<int>(ncols)) {
		throw std::out_of_range("Column index out of range.");
	}
	if (colIndex < -1) {
		throw std::invalid_argument("mat::getCol: Invalid column index.");
	}

	if(colIndex == -1){
		colIndex = ncols-1;
	}

	vec column(nrows);
	for (unsigned int row = 0; row < nrows; ++row) {
		column(row) = matrix[colIndex * nrows + row];
	}

	return column;
}


void mat::setCol(const vec& colVec, unsigned int colIndex) {
	if (colIndex >= ncols) {
		throw std::out_of_range("Column index out of range.");
	}

	if (colVec.getSize() != nrows) {
		throw std::invalid_argument("Vector size does not match the number of rows.");
	}

	for (unsigned int row = 0; row < nrows; ++row) {
		matrix[colIndex * nrows + row] = colVec(row);
	}
}

vec mat::getRow(int rowIndex) const {
	if (rowIndex >= static_cast<int>(nrows)) {
		throw std::out_of_range("mat::getRow: Row index out of range.");
	}
	if (rowIndex < -1) {
		throw std::invalid_argument("mat::getRow: Invalid row index.");
	}

	if(rowIndex == -1){
		rowIndex = nrows-1;
	}

	vec row(ncols);
	for (unsigned int col = 0; col < ncols; ++col) {
		row(col) = matrix[col * nrows + rowIndex];
	}

	return row;
}


void mat::setRow(const vec& rowVec, unsigned int rowIndex) {
	if (rowIndex >= nrows) {
		throw std::out_of_range("Row index out of range.");
	}

	if (rowVec.getSize() != ncols) {
		throw std::invalid_argument("Vector size does not match the number of columns.");
	}

	for (unsigned int col = 0; col < ncols; ++col) {
		matrix[col * nrows + rowIndex] = rowVec(col);
	}
}


vec mat::diag() const {
	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square to extract the diagonal.");
	}
	if (isEmpty()) {
		throw std::invalid_argument("Matrix must be non empty to extract the diagonal.");
	}

	vec diagonal(nrows);
	for (unsigned int i = 0; i < nrows; ++i) {
		diagonal(i) = matrix[i * (nrows + 1)];
	}

	return diagonal;
}


bool mat::isSymmetric(double tolerance) const {

	if (tolerance < 0.0) {
		throw std::invalid_argument("mat::isSymmetric: Tolerance cannot be negative.");
	}

	if (nrows != ncols) {
		return false; // A non-square matrix cannot be symmetric
	}

	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = i + 1; j < ncols; ++j) {
			if (std::fabs(matrix[j * nrows + i] - matrix[i * nrows + j]) > tolerance) {
				return false;
			}
		}
	}

	return true;
}


bool mat::isEqual(const mat& other, double tolerance) const {
	if (nrows != other.nrows || ncols != other.ncols) {
		return false; // Matrices with different dimensions cannot be equal
	}

	for (unsigned int i = 0; i < nrows * ncols; ++i) {
		if (std::fabs(matrix[i] - other.matrix[i]) > tolerance) {
			return false; // Found a difference greater than the tolerance
		}
	}

	return true; // All elements are within the tolerance
}


void mat::generateRandomPositiveDefinite() {
	if (nrows != ncols) {
		throw std::invalid_argument("mat::generateRandomSymmetric: Matrix must be square.");
	}
	if (nrows*ncols == 0) {
		throw std::invalid_argument("mat::generateRandomSymmetric: Matrix is empty.");
	}

	mat temp(nrows, ncols);
	temp.fillRandom();


	// Compute A^T * A manually
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < nrows; ++j) {
			double sum = 0.0;
			for (unsigned int k = 0; k < ncols; ++k) {
				sum += temp.matrix[k * nrows + i] * temp.matrix[k * nrows + j];
			}
			matrix[j * nrows + i] = sum;
		}
	}
}

void mat::generateRandomSymmetric() {

	if (nrows != ncols) {
		throw std::invalid_argument("mat::generateRandomSymmetric: Matrix must be square.");
	}
	if (nrows*ncols == 0) {
		throw std::invalid_argument("mat::generateRandomSymmetric: Matrix is empty.");
	}
	fillRandom();

	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = i + 1; j < ncols; ++j) {
			matrix[j * nrows + i] = matrix[i * nrows + j];

		}
	}

}

mat mat::computeCorrelationMatrixExponential(const vec &theta, const vec &gamma) const {

	if(theta.getSize() != gamma.getSize()){
		throw std::invalid_argument("mat::computeCorrelationMatrixExponential: theta and gamma must have the same dimension.");
	}

	if(ncols != theta.getSize()){
		throw std::invalid_argument("mat::computeCorrelationMatrixExponential: number of columns of the input matrix and theta must be the same.");

	}

#ifdef OPENBLAS
	return computeCorrelationMatrixExponentialOpenBlas(theta, gamma);
#else
	return computeCorrelationMatrixExponentialNaive(theta, gamma);
#endif

}


mat mat::computeCorrelationMatrixExponentialNaive(const vec &theta, const vec &gamma) const {

	vector<vec> rowsOfX;

	for(unsigned int i=0;i<nrows; i++){
		rowsOfX.push_back(this->getRow(i));
	}

	mat R(nrows, nrows);  // Create the correlation matrix

	// Set the diagonal elements to 1.0 (correlation of each row with itself)
	for (unsigned int i = 0; i < nrows; ++i) {
		R.matrix[i * nrows + i] = 1.0;
	}

	// Compute the correlation for each pair of rows
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			double corr = vec::computeExponentialCorrelationNaive(rowsOfX[i].getPointer(),
					rowsOfX[j].getPointer(),
					theta.getPointer(),
					gamma.getPointer(),
					ncols);
			R.matrix[i * nrows + j] = corr;  // Fill lower triangle
			R.matrix[j * nrows + i] = corr;  // Fill symmetric element in upper triangle
		}
	}

	return R;
}

#ifdef OPENBLAS

mat mat::computeCorrelationMatrixExponentialOpenBlas(const vec &theta, const vec &gamma) const {

	vector<vec> rowsOfX;

	for(unsigned int i=0;i<nrows; i++){
		rowsOfX.push_back(this->getRow(i));
	}

	mat R(nrows, nrows);  // Create the correlation matrix

	// Set the diagonal elements to 1.0 (correlation of each row with itself)
	for (unsigned int i = 0; i < nrows; ++i) {
		R.matrix[i * nrows + i] = 1.0;
	}

	// Compute the correlation for each pair of rows
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			double corr = vec::computeExponentialCorrelationOpenBlas(rowsOfX[i].getPointer(),
					rowsOfX[j].getPointer(),
					theta.getPointer(),
					gamma.getPointer(),
					ncols);
			R.matrix[i * nrows + j] = corr;  // Fill lower triangle
			R.matrix[j * nrows + i] = corr;  // Fill symmetric element in upper triangle
		}
	}

	return R;
}


#endif


mat mat::computeCorrelationMatrixGaussian(const vec &theta) const {


	if(ncols != theta.getSize()){
		throw std::invalid_argument("mat::computeCorrelationMatrixExponential: number of columns of the input matrix and theta must be the same.");

	}

#ifdef OPENBLAS
	return computeCorrelationMatrixGaussianOpenBlas(theta);
#else
	return computeCorrelationMatrixGaussianNaive(theta);
#endif

}


mat mat::computeCorrelationMatrixGaussianNaive(const vec &theta) const {

	vector<vec> rowsOfX;

	for(unsigned int i=0;i<nrows; i++){
		rowsOfX.push_back(this->getRow(i));
	}

	mat R(nrows, nrows);  // Create the correlation matrix

	// Set the diagonal elements to 1.0 (correlation of each row with itself)
	for (unsigned int i = 0; i < nrows; ++i) {
		R.matrix[i * nrows + i] = 1.0;
	}

	// Compute the correlation for each pair of rows
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			double corr = vec::computeGaussianCorrelationNaive(
					rowsOfX[i].getPointer(),
					rowsOfX[j].getPointer(),
					theta.getPointer(),
					ncols);
			R.matrix[i * nrows + j] = corr;  // Fill lower triangle
			R.matrix[j * nrows + i] = corr;  // Fill symmetric element in upper triangle
		}
	}

	return R;
}

#ifdef OPENBLAS

mat mat::computeCorrelationMatrixGaussianOpenBlas(const vec &theta) const {

	vector<vec> rowsOfX;

	for(unsigned int i=0;i<nrows; i++){
		rowsOfX.push_back(this->getRow(i));
	}

	mat R(nrows, nrows);  // Create the correlation matrix

	// Set the diagonal elements to 1.0 (correlation of each row with itself)
	for (unsigned int i = 0; i < nrows; ++i) {
		R.matrix[i * nrows + i] = 1.0;
	}

	// Compute the correlation for each pair of rows
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			double corr = vec::computeGaussianCorrelationOpenBlas(rowsOfX[i].getPointer(),
					rowsOfX[j].getPointer(),
					theta.getPointer(),
					ncols);
			R.matrix[i * nrows + j] = corr;  // Fill lower triangle
			R.matrix[j * nrows + i] = corr;  // Fill symmetric element in upper triangle
		}
	}

	return R;
}


#endif




void mat::generateRandomCorrelationMatrix() {

	if (nrows != ncols) {
		throw std::invalid_argument("mat::generateRandomSymmetric: Matrix must be square.");
	}

	mat X(nrows,nrows);
	X.fillRandom();
	vec theta(nrows,1.0);

	vector<vec> rowsOfX;

	for(unsigned int i=0;i<nrows; i++){
		rowsOfX.push_back(X.getRow(i));
	}
	for(unsigned int i=0;i<nrows; i++){
		matrix[i*nrows+i] = 1.0;
	}

	for(unsigned int i=0;i<nrows; i++){
		for(unsigned int j=0;j<i; j++){
			double corr = vec::computeGaussianCorrelation(rowsOfX[i].getPointer(),rowsOfX[j].getPointer(),theta.getPointer(), nrows);
			matrix[i*nrows+j] = corr;
			matrix[i+nrows*j] = corr;
		}
	}
}



void mat::print(std::string msg, int p) const {

	std::cout<<"\n";
	if(!msg.empty()){
		std::cout<<msg<<"\n";
	}
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < ncols; ++j) {
			std::cout << std::setw(10) << std::setprecision(p) << std::fixed << (*this)(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}


void mat::saveAsCSV(const std::string& filename, int precision, const std::vector<std::string>& header) const {
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
		throw std::runtime_error("Unable to open file for writing: " + filename);
	}

	// Set the precision for the output
	outFile << std::fixed << std::setprecision(precision);

	try {
		// Write header if provided
		if (!header.empty()) {
			if (header.size() != ncols) {
				throw std::invalid_argument("Header size does not match the number of columns.");
			}
			for (size_t i = 0; i < header.size(); ++i) {
				outFile << header[i];
				if (i < header.size() - 1) {
					outFile << ",";
				}
			}
			outFile << "\n";
		}

		// Write matrix data
		for (unsigned int row = 0; row < nrows; ++row) {
			for (unsigned int col = 0; col < ncols; ++col) {
				outFile << matrix[col * nrows + row];
				if (col < ncols - 1) {
					outFile << ",";
				}
			}
			outFile << "\n";
		}

		if (outFile.fail()) {
			throw std::runtime_error("Failed to write to file: " + filename);
		}
	} catch (...) {
		outFile.close(); // Ensure file is closed in case of an exception
		throw;           // Re-throw the caught exception
	}

	outFile.close();
}



void mat::readFromCSV(const std::string& filename, bool hasHeader) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file: " + filename);
	}

	std::string line;
	std::vector<std::vector<double>> data;

	// Only skip header if `hasHeader` is true and file is not empty
	if (hasHeader){
		std::string header;
		std::getline(file, header);
		//		std::cout<<"header = "<< header <<"\n";
	}

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		//		std::cout<<"line = "<<line <<"\n";
		std::string value;
		std::vector<double> row;

		while (std::getline(ss, value, ',')) {
			row.push_back(std::stod(value));
		}

		data.push_back(row);
	}

	file.close();

	if (data.empty()) {
		throw std::runtime_error("No data found in file: " + filename);
	}

	// Determine the number of rows and columns
	nrows = data.size();
	ncols = data[0].size();

	// Allocate memory for the matrix
	delete[] matrix;  // Free the old matrix if it exists
	matrix = new double[nrows * ncols];

	// Fill the matrix with data from the vector
	for (unsigned int i = 0; i < nrows; ++i) {
		if (data[i].size() != static_cast<size_t>(ncols)) {
			throw std::runtime_error("Inconsistent number of columns in row " + std::to_string(i));
		}
		for (unsigned int j = 0; j < ncols; ++j) {
			matrix[j * nrows + i] = data[i][j];  // Column-major order
		}
	}
}



void mat::printPivot(const std::vector<int>& pivot) {
	std::cout << "Pivot Indices:\n";
	for (size_t i = 0; i < pivot.size(); ++i) {
		std::cout << pivot[i] << " "; // Print 1-based indices
	}
	std::cout << "\n";
}


vec mat::matVecProduct(const vec& v) const {

	if (v.getSize() != ncols) {
		throw std::invalid_argument("Matrix-vector product: Vector size must match the number of columns in the matrix.");
	}

#ifdef OPENBLAS
	return matVecProductOpenBlas(v);
#else
	return matVecProductNaive(v);
#endif

}

#ifdef OPENBLAS
vec mat::matVecProductOpenBlas(const vec& v) const {

	vec result(nrows);

	// Use cblas_dgemv to perform matrix-vector multiplication
	cblas_dgemv(CblasColMajor, CblasNoTrans, nrows, ncols, 1.0, matrix, nrows, v.getPointer(), 1, 0.0, result.getPointer(), 1);

	return result;
}
#endif


vec mat::matVecProductNaive(const vec& v) const {

	vec result(nrows);

	// Perform the matrix-vector multiplication using column-major order
	for (unsigned int i = 0; i < nrows; ++i) {
		result(i) = 0.0;
		for (unsigned int j = 0; j < ncols; ++j) {
			result(i) += matrix[j * nrows + i] * v(j);
		}
	}

	return result;
}

mat mat::transpose() const {

	return transposeNaive();

}


mat mat::transposeNaive() const {
	mat result(ncols, nrows);
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < ncols; ++j) {
			result.matrix[i * ncols + j] = matrix[j * nrows + i];
		}
	}
	return result;
}

mat mat::cholesky(int &result) const {
	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square for Cholesky decomposition.");
	}


#ifdef OPENBLAS
	return choleskyOpenBlas(result);
#else
	return choleskyNaive(result);
#endif
}

#ifdef OPENBLAS
mat mat::choleskyOpenBlas(int& ret) const {

	mat result(*this); // Create a copy of the matrix

	int n = nrows;

	// Perform Cholesky decomposition
	int info;
	dpotrf_("L", &n, result.matrix, &n, &info);
	if (info > 0) {
		ret = -1;
		return result;
	}

	// Zero out the upper triangular part
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = i + 1; j < ncols; ++j) {
			result.matrix[j * nrows + i] = 0.0;
		}
	}
	ret = 1;
	return result;
}
#endif

mat mat::choleskyNaive(int& ret) const {

	mat L(nrows, ncols);
	unsigned int n = nrows;

	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j <= i; j++) {
			double sum = 0;
			for (unsigned int k = 0; k < j; k++) {
				sum += L.matrix[i + k * n] * L.matrix[j + k * n]; // Accessing elements in column-major order
			}

			if (i == j) {
				double fac = matrix[i + i * n] - sum;
				if(fac < 0) {
					ret = -1;
					return L;
				}
				L.matrix[i + j * n] = std::sqrt(fac); // Accessing elements in column-major order
			} else {
				L.matrix[i + j * n] = (1.0 / L.matrix[j + j * n] * (matrix[i + j * n] - sum)); // Accessing elements in column-major order
			}
		}
	}

	// Zero out the upper triangular part
	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = i + 1; j < ncols; ++j) {
			L.matrix[j * nrows + i] = 0.0;
		}
	}
	ret = 1;

	return L;

}


// Matrix addition
mat mat::operator+(const mat& other) const {
	if (nrows != other.nrows || ncols != other.ncols) {
		throw std::invalid_argument("Matrix dimensions must agree for addition.");
	}
	mat result(nrows, ncols);
	for (unsigned int i = 0; i < nrows*ncols; ++i) {
		result.matrix[i] = matrix[i] + other.matrix[i];
	}
	return result;
}

// Matrix subtraction
mat mat::operator-(const mat& other) const {
	if (nrows != other.nrows || ncols != other.ncols) {
		throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
	}
	mat result(nrows, ncols);
	for (unsigned int i = 0; i < nrows*ncols; ++i) {
		result.matrix[i] = matrix[i] - other.matrix[i];
	}
	return result;
}


mat mat::operator*(const mat& other) const {
	if (ncols != other.nrows) {
		throw std::invalid_argument("Matrix dimensions must agree for multiplication.");
	}

	return matMatProduct(other);
}

mat mat::operator*(double scalar) const {
	mat result(nrows, ncols);
	for (unsigned int i = 0; i < nrows*ncols; ++i) {
		result.matrix[i] = scalar*matrix[i];
	}
	return result;
}






mat mat::matMatProduct(const mat& other) const {

	if (ncols != other.nrows) {
		throw std::invalid_argument("Matrix dimensions must agree.");
	}


#ifdef OPENBLAS

	return matMatProductOpenBlas(other);
#else

	return matMatProductNaive(other);
#endif
}


mat mat::matMatProductNaive(const mat& other) const {

	mat result(nrows, other.ncols);

	for (unsigned int i = 0; i < nrows; ++i) {
		for (unsigned int j = 0; j < other.ncols; ++j) {
			double sum = 0.0;
			for (unsigned int k = 0; k < ncols; ++k) {
				sum += matrix[k * nrows + i] * other.matrix[j * other.nrows + k];
			}
			result.matrix[j * nrows + i] = sum;
		}
	}

	return result;
}
#ifdef OPENBLAS
mat mat::matMatProductOpenBlas(const mat& other) const {

	mat result(nrows, other.ncols);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, other.ncols, ncols, 1.0, matrix, nrows, other.matrix, other.nrows, 0.0, result.matrix, nrows);

	return result;
}
#endif


vec mat::solveCholesky(const vec& b) const {

	if (nrows != ncols || nrows != b.getSize()) {
		throw std::invalid_argument("Invalid dimensions for matrix and vector in solve function.");
	}

#ifdef OPENBLAS
	return solveCholeskyOpenBlas(b);
#else
	return solveCholeskyNaive(b);
#endif

}

#ifdef OPENBLAS
vec mat::solveCholeskyOpenBlas(const vec& b) const {

	int n = nrows;
	vec y(b);
	//    y.print("y");


	// Forward substitution to solve Ly = b
	cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, matrix, n, y.getPointer(), 1);

	vec x(y);
	// Backward substitution to solve L^T x = y
	cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, matrix, n, x.getPointer(), 1);

	return x;
}

#endif

vec mat::solveCholeskyNaive(const vec& b) const {

	int n = nrows;

	vec y(n);
	vec x(n);

	// Step 1: Forward substitution to solve Ly = b
	for (int i = 0; i < n; ++i) {
		double sum = 0.0;
		for (int j = 0; j < i; ++j) {
			sum += matrix[j * n + i] * y(j); // Access in column-major format
		}
		y(i) = (b(i) - sum) / matrix[i * n + i]; // Access in column-major format
	}

	// Step 2: Backward substitution to solve L^T x = y
	for (int i = n - 1; i >= 0; --i) {
		double sum = 0.0;
		for (int j = i + 1; j < n; ++j) {
			sum += matrix[i * n + j] * x(j); // Access in column-major format
		}
		x(i) = (y(i) - sum) / matrix[i * n + i]; // Access in column-major format
	}

	return x;
}


void mat::addEpsilonToDiagonal(double epsilon) {
	unsigned int min_dim = std::min(nrows, ncols);
	for (unsigned int i = 0; i < min_dim; ++i) {
		matrix[i * nrows + i] += epsilon;
	}
}


mat mat::invert() const{
	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square to be inverted.");
	}

#ifdef OPENBLAS
	return invertOpenBlas();
#else
	return invertNaive();
#endif

}


mat mat::invertNaive() const {

	unsigned int n = nrows;
	mat inverse(n, n);

	// Create a copy of the matrix and perform LU decomposition
	mat LU(*this);
	std::vector<int> pivot(n);
	int info;

	LU = LU.luNaive(pivot.data(), info);
	if (info < 0) {
		throw std::runtime_error("Matrix is singular and cannot be inverted.");
	}

	// Solve for each column of the identity matrix
	for (unsigned int i = 0; i < n; ++i) {
		vec e(n, 0.0);
		e(i) = 1.0;
		vec col = LU.solveLUNaive(pivot.data(), e);
		for (unsigned int j = 0; j < n; ++j) {
			inverse(j, i) = col(j);
		}
	}

	return inverse;
}


#ifdef OPENBLAS
mat mat::invertOpenBlas() const {

	int n = nrows;
	mat inverse(*this); // Copy the original matrix
	std::vector<int> pivot(n);
	int info;

	// LU decomposition
	dgetrf_(&n, &n, inverse.getPointer(), &n, pivot.data(), &info);
	if (info != 0) {
		throw std::runtime_error("Matrix is singular and cannot be inverted.");
	}

	// Allocate workspace
	int lwork = n * 64; // A common choice for lwork is n * some multiple, such as 64
	std::vector<double> work(lwork);

	// Matrix inversion
	dgetri_(&n, inverse.getPointer(), &n, pivot.data(), work.data(), &lwork, &info);
	if (info != 0) {
		throw std::runtime_error("Matrix inversion failed.");
	}

	return inverse;
}
#endif


mat mat::lu(int* pivot, int &return_value) const{

	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square for LU decomposition.");
	}


#ifdef OPENBLAS
	return luOpenBlas(pivot,return_value);
#else
	return luNaive(pivot,return_value);
#endif

}


vec mat::solveLU(int *pivot, const vec& b) const {

#ifdef OPENBLAS
	return solveLUOpenBlas(pivot,b);
#else
	return solveLUNaive(pivot,b);
#endif


}


#ifdef OPENBLAS
mat mat::luOpenBlas(int* pivot,int &return_value) const{


	int n = nrows;
	double *LUMatrix = new double[n*n];

	for(unsigned int i=0; i<nrows*nrows; i++){

		LUMatrix[i] = matrix[i];
	}

	// Perform LU decomposition using dgetrf
	int info;
	dgetrf_(&n, &n, LUMatrix, &n, pivot, &info);
	if (info < 0) {
		throw std::runtime_error("Argument " + std::to_string(-info) + " had an illegal value.");
	} else if (info > 0) {
		return_value = -1;
		mat emptyMat;
		return emptyMat;
	}

	mat result;
	result.matrix = LUMatrix;
	result.ncols = ncols;
	result.nrows = nrows;
	return_value = 1;
	return result;

}


vec mat::solveLUOpenBlas(int *pivot, const vec& b) const {
	if (nrows!= ncols || nrows != b.getSize()) {
		std::cout<< "nrows = " << nrows << "\n";
		std::cout<< "nrows = " << ncols << "\n";
		std::cout<< "b.getSize() = " << b.getSize() << "\n";
		throw std::invalid_argument("mat::solveLUOpenBlas: Invalid dimensions for matrices or vector in solve function.");
	}

	int n = nrows;
	int nrhs = 1;  // Number of right-hand sides, i.e., the number of columns of B
	int info;


	//	mat::printPivot(pivot);

	// Copy the right-hand side vector b
	vec x(b);

	// Solve the system using dgetrs
	char trans = 'N';  // 'N' means solve A * X = B
	dgetrs_(&trans, &n, &nrhs, matrix, &n, pivot, x.getPointer(), &n, &info);
	if (info < 0) {
		throw std::runtime_error("Argument " + std::to_string(-info) + " had an illegal value.");
	}


	return x;
}
#endif

mat mat::luNaive(int *pivot, int & return_value) const {
	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square for LU decomposition.");
	}

	if (pivot == nullptr) {
		throw std::invalid_argument("mat::luNaive: Null pointer for pivot.");
	}

	int n = nrows;

	double *LU = mat::lu_decomposition(matrix, n, pivot, &return_value);


	mat result;
	result.matrix = LU;
	result.ncols = ncols;
	result.nrows = nrows;

	return result;

}


vec mat::solveLUNaive(int *pivot, const vec& b) const {
	// this function assumes that matrix saves L and U, LU decomposition is already done
	if (nrows != ncols || nrows != b.getSize()) {
		throw std::invalid_argument("Matrix must be square and the vector size must match the matrix size.");
	}

	int n = nrows;

	vec x(n);

	lu_solve(matrix, pivot, b.getPointer(), x.getPointer(), n);

	return x;
}




// Function to print the matrix
void mat::print_matrix(const double *a, int n) {
	printf("\n");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			printf("%f ", a[i + j * n]);
		}
		printf("\n");
	}
	printf("\n");
}


void column_major_to_row_major(double* col_major, double* row_major, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// Convert column-major index to row-major index
			row_major[i * n + j] = col_major[j * n + i];
		}
	}
}



void mat::print_permutation(int *p, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", p[i]);
	}
	printf("\n");
}

double* mat::lu_decomposition(double *a, int n, int *p, int * return_value) {
	double *lu = new double[n*n];
	if (lu == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Initialize the permutation array and copy 'a' to 'lu'
	for (int i = 0; i < n; i++) {
		p[i] = i;
		for (int j = 0; j < n; j++) {
			lu[i + j * n] = a[i + j * n];
		}
	}

	for (int k = 0; k < n; k++) {
		// Find the pivot row
		int max_row = k;
		for (int i = k + 1; i < n; i++) {
			if (fabs(lu[i + k * n]) > fabs(lu[max_row + k * n])) {
				max_row = i;
			}
		}

		// Swap the pivot row with the current row
		if (max_row != k) {
			//        	std::cout<<"swap = " << k << " and " << max_row <<"\n";
			for (int j = 0; j < n; j++) {
				double temp = lu[k + j * n];
				lu[k + j * n] = lu[max_row + j * n];
				lu[max_row + j * n] = temp;
			}
			int temp_p = p[k];
			p[k] = p[max_row];
			p[max_row] = temp_p;
		}
		//        print_permutation(p, n);

		// Check for zero pivot
		if (fabs(lu[k + k * n]) < 10E-10) {

			delete[] lu;
			*return_value = -1;
			return nullptr;
		}

		// Compute the multipliers and store them in the lower part of lu
		for (int i = k + 1; i < n; i++) {
			lu[i + k * n] /= lu[k + k * n];
		}

		// Update the remaining submatrix
		for (int j = k + 1; j < n; j++) {
			for (int i = k + 1; i < n; i++) {
				lu[i + j * n] -= lu[i + k * n] * lu[k + j * n];
			}
		}
	}
	*return_value = 1;
	return lu;
}
void mat::lu_solve(double *LU, int *p, double *b, double *x, int n) {
	double *y = new double[n];
	if (y == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Forward substitution to solve Ly = Pb
	forward_substitution(n, LU, p, b, y);

	// Backward substitution to solve Ux = y
	backward_substitution(n, LU, y, x);

	delete[] y;

}
void mat::forward_substitution(int n, double *LU, int *p, double *b, double *y) {
	// Apply the permutation to b (i.e., compute Pb)
	for (int i = 0; i < n; i++) {
		y[i] = b[p[i]];
	}

	// Solve Ly = Pb, where L has unit diagonal elements
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			y[i] -= LU[i + j * n] * y[j];
		}
		// y[i] /= 1; // No division needed since L has unit diagonal
	}
}

void mat::backward_substitution(int n, double *LU, double *y, double *x) {
	// Solve Ux = y
	for (int i = n - 1; i >= 0; i--) {
		x[i] = y[i];
		for (int j = i + 1; j < n; j++) {
			x[i] -= LU[i + j * n] * x[j];
		}
		x[i] /= LU[i + i * n];
	}
}

double* mat::matvecmul(double *a, double *b, int n) {
	double *result = new double[n];

	if (result == nullptr) {
		throw std::runtime_error("mat::matvecmul: Memory allocation failed.");
	}

	// Initialize the result vector to zero
	for (int i = 0; i < n; i++) {
		result[i] = 0.0;
	}

	// Perform matrix-vector multiplication
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			result[i] += a[i + j * n] * b[j];
		}
	}

	return result;
}

void mat::matmatmul(double *A, double *B, double *C, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			C[i + j * n] = 0.0;
			for (int k = 0; k < n; k++) {
				C[i + j * n] += A[i + k * n] * B[k + j * n];
			}
		}
	}
}


void mat::shuffleRows() {
	std::random_device rd; // Seed for the random number generator
	std::mt19937 g(rd());  // Mersenne Twister random number generator

	std::vector<int> row_indices(nrows);
	for (unsigned int i = 0; i < nrows; ++i) {
		row_indices[i] = i;
	}

	// Shuffle the row indices
	std::shuffle(row_indices.begin(), row_indices.end(), g);

	// Create a new matrix to hold the shuffled rows
	double* shuffled_matrix = new double[nrows * ncols];
	if (shuffled_matrix == nullptr) {
		throw std::runtime_error("Memory allocation failed.");
	}

	// Copy rows according to the shuffled order
	for (unsigned int new_row = 0; new_row < nrows; ++new_row) {
		int old_row = row_indices[new_row];
		for (unsigned int col = 0; col < ncols; ++col) {
			shuffled_matrix[col * nrows + new_row] = matrix[col * nrows + old_row];
		}
	}

	// Replace the old matrix with the shuffled one
	delete[] matrix;
	matrix = shuffled_matrix;
}


bool mat::isRowInMatrix(const vec& v, double tolerance) const {
	if (v.getSize() != ncols) {
		return false; // The vector does not match the number of columns
	}

	for (unsigned int i = 0; i < nrows; ++i) {
		bool isRowMatch = true;
		for (unsigned int j = 0; j < ncols; ++j) {
			if (std::fabs(matrix[j * nrows + i] - v(j)) > tolerance) {
				isRowMatch = false;
				break;
			}
		}
		if (isRowMatch) {
			return true; // Found a matching row
		}
	}

	return false; // No matching row found
}


int mat::findRowIndex(const vec& v, double tolerance ) const {

	if (v.getSize() != ncols) {
		throw std::invalid_argument("Vector size must match the number of columns in the matrix.");
	}

	for (unsigned int i = 0; i < nrows; ++i) {
		bool isRowMatch = true;
		for (unsigned int j = 0; j < ncols; ++j) {

			double diff =std::fabs( matrix[j * nrows + i] - v(j) );
			if (diff > tolerance) {
				isRowMatch = false;
				break;
			}
		}
		if (isRowMatch) {
			return i; // Found a matching row, return its index
		}
	}

	return -1; // No matching row found
}


mat mat::normalizeMatrix(vec xmin, vec xmax) const{

	unsigned int dim = xmin.getSize();

	if( dim != xmax.getSize() ){

		throw std::invalid_argument("mat::normalizeMatrix: xmin and xmax must have the same size.");
	}

	if(dim != ncols){

		throw std::invalid_argument("mat::normalizeMatrix: number of columns does not match with xmin and xmax.");
	}

	for(unsigned int i=0; i<dim; i++){

		if(xmin(i) >= xmax(i)){
			throw std::invalid_argument("mat::normalizeMatrix: xmax(i) must be larger than xmin(i).");

		}

	}
	mat matrixOut(nrows, ncols);

	for(unsigned int i=0; i<nrows;i++){

		for(unsigned int j=0; j<dim;j++){

			unsigned int idx = j * nrows + i;
			double deltax = xmax(j) - xmin(j);
			matrixOut(i,j)  = (matrix[idx] - xmin(j))/deltax;
		}
	}

	return matrixOut*(1.0/dim);

}

mat mat::normalizeMatrix(double xmin, double xmax) const{

	if(xmin >= xmax ){
		throw std::invalid_argument("mat::normalizeMatrix: xmax(i) must be larger than xmin(i).");

	}

	vec minx(ncols);
	minx.fill(xmin);

	vec maxx(ncols);
	maxx.fill(xmax);

	return normalizeMatrix(minx,maxx);

}


mat mat::concatenateRowWise(const mat& other) const {

	if (nrows == 0) {
		return other;  // If this matrix is empty, return the other matrix
	}
	if (other.nrows == 0) {
		return *this;  // If the other matrix is empty, return this matrix
	}
	// Check if the number of columns matches
	if (ncols != other.ncols) {
		throw std::invalid_argument("Matrices must have the same number of columns to concatenate row-wise.");
	}

	// Calculate the total number of rows for the new matrix
	int newRows = nrows + other.nrows;

	// Create a new matrix with the combined number of rows and the same number of columns
	mat result(newRows, ncols);

	// Copy the data from the first matrix
	for (unsigned int col = 0; col < ncols; ++col) {
		for (unsigned int row = 0; row < nrows; ++row) {
			result(row, col) = matrix[col * nrows + row];
		}
	}

	// Copy the data from the second matrix
	for (unsigned int col = 0; col < ncols; ++col) {
		for (unsigned int row = 0; row < other.nrows; ++row) {
			result(nrows + row, col) = other.matrix[col * other.nrows + row];
		}
	}

	return result;
}


mat mat::concatenateColumnWise(const mat& other) const {
	// Handle the case where one of the matrices is empty
	if (ncols == 0) {
		return other;  // If this matrix is empty, return the other matrix
	}
	if (other.ncols == 0) {
		return *this;  // If the other matrix is empty, return this matrix
	}

	// Check if the number of rows matches
	if (nrows != other.nrows) {
		throw std::invalid_argument("Matrices must have the same number of rows to concatenate column-wise.");
	}

	// Calculate the total number of columns for the new matrix
	int newCols = ncols + other.ncols;

	// Create a new matrix with the combined number of columns and the same number of rows
	mat result(nrows, newCols);

	// Copy the data from the first matrix
	for (unsigned int col = 0; col < ncols; ++col) {
		for (unsigned int row = 0; row < nrows; ++row) {
			result(row, col) = matrix[col * nrows + row];
		}
	}

	// Copy the data from the second matrix
	for (unsigned int col = 0; col < other.ncols; ++col) {
		for (unsigned int row = 0; row < nrows; ++row) {
			result(row, ncols + col) = other.matrix[col * other.nrows + row];
		}
	}

	return result;
}


#ifdef OPENBLAS
vec mat::solve(const vec& b) const {
	if (nrows != ncols) {
		throw std::invalid_argument("Matrix must be square to solve Ax = b.");
	}

	if (nrows != b.getSize()) {
		throw std::invalid_argument("The size of vector b must match the number of rows of the matrix.");
	}

	int n = nrows;  // Matrix size (n x n)
	int nrhs = 1;   // Number of right-hand sides (i.e., the number of columns in b)
	int info;       // Info variable to capture success or failure
	std::vector<int> ipiv(n);  // Array to hold pivot indices

	// Copy matrix and vector b to modify during the solution process
	mat A_copy(*this);  // Copy of matrix (as dgesv overwrites it)
	vec x(b);           // Copy of the right-hand side vector (as dgesv overwrites it)

	// Call OpenBLAS LAPACK routine dgesv to solve Ax = b using LU decomposition
	dgesv_(&n, &nrhs, A_copy.getPointer(), &n, ipiv.data(), x.getPointer(), &n, &info);

	if (info < 0) {
		throw std::runtime_error("The " + std::to_string(-info) + "-th argument had an illegal value.");
	} else if (info > 0) {
		throw std::runtime_error("The matrix is singular and the solution could not be computed.");
	}

	// The solution is now in x
	return x;
}
#else
vec mat::solve(const vec& b) const {
	throw std::runtime_error("OpenBLAS is not enabled. Cannot solve the system.");
}
#endif



} /* namespace Rodop */
