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
#ifndef MAT_H_
#define MAT_H_
#include "vector.hpp"
#include <vector>
namespace Rodop {

class mat {
private:
	unsigned int nrows;
	unsigned int ncols;
	double* matrix;




public:
	// Constructors
	mat();
	mat(unsigned int nRows, unsigned int nCols);

	mat(unsigned int r, unsigned int c, double val);

	// Copy constructor
	mat(const mat& other);

	// Copy assignment operator
	mat& operator=(const mat& other);

	// Move constructor
	mat(mat&& other) noexcept;

	// Move assignment operator
	mat& operator=(mat&& other) noexcept;

	// Destructor
	~mat();

	double * getPointer() const;
	void reset();
	void resize(unsigned int newRows, unsigned int newCols);

	unsigned int getNRows() const { return nrows; }
	unsigned int getNCols() const { return ncols; }
	unsigned int getSize() const { return nrows * ncols; }
//
//	// Matrix addition
	mat operator+(const mat& other) const;
//
//	// Matrix subtraction
	mat operator-(const mat& other) const;
//
//	// Matrix multiplication
	mat operator*(const mat& other) const;
	mat operator*(double scalar) const;

	double& operator()(unsigned int i, unsigned int j);
	const double& operator()(unsigned int i, unsigned int j) const;

	// Display matrix
	void print(std::string msg = "", int precision = 4) const;
	void display() const { print(); }
	static	void printPivot(const std::vector<int>& pivot);

	void saveAsCSV(const std::string& filename, int precision = 6, const std::vector<std::string>& header = {}) const;
	void readFromCSV(const std::string& filename, bool hasHeader = false);

	mat submat(unsigned int startRow, unsigned int startCol, unsigned int numRows, unsigned int numCols) const;

	bool isSymmetric(double tolerance = 1e-9) const;
	bool isEmpty()const{
		if(nrows == 0 || ncols == 0) return true;
		else return false;
	}
    bool isEqual(const mat& other, double tolerance = 1e-9) const;

	void fill(double value);
	void fillRandom();
	void fillRandom(double a, double b);
	void fillRandom(const vec& lb, const vec& ub);
	void fillRandom(const std::vector<double>& lb, const std::vector<double>& ub);
	void fillRandomLHS(const vec& lb, const vec& ub);
	void fillRandomLHS(const std::vector<double>& lb, const std::vector<double>& ub);

	void eye();

	void generateRandomPositiveDefinite();
	void generateRandomSymmetric();

	mat computeCorrelationMatrixExponential(const vec &theta, const vec &gamma) const;
	mat computeCorrelationMatrixExponentialNaive(const vec &theta, const vec &gamma) const;
#ifdef OPENBLAS
	mat computeCorrelationMatrixExponentialOpenBlas(const vec &theta, const vec &gamma) const;
#endif

	mat computeCorrelationMatrixGaussian(const vec &theta) const;
	mat computeCorrelationMatrixGaussianNaive(const vec &theta) const;
#ifdef OPENBLAS
	mat computeCorrelationMatrixGaussianOpenBlas(const vec &theta) const;
#endif


	void generateRandomCorrelationMatrix();

	/* Row and column operations */
	void deleteRow(int rowIndex);
	void deleteRows(const std::vector<int>& rowIndices);
	void addRow(const vec& rowVec, int position = -1);
	void addColumn(const vec& colVec, int position = -1);
	vec getCol(int colIndex) const;
	vec getRow(int rowIndex) const;
	void setRow(const vec& rowVec, unsigned int rowIndex);
	void setCol(const vec& colVec, unsigned int colIndex);
	void shuffleRows();
	bool isRowInMatrix(const vec& v, double tolerance = 10E-9) const;
	int findRowIndex(const vec& v, double tolerance = 10E-9) const;

	vec diag() const;

	mat transpose() const;
	mat transposeNaive() const;

	vec matVecProduct(const vec& v) const;
	vec matVecProductNaive(const vec& v) const;
	vec matVecProductOpenBlas(const vec& v) const;

	mat matMatProduct(const mat& other) const;
	mat matMatProductOpenBlas(const mat& other) const;
	mat matMatProductNaive(const mat& other) const;


	mat invert() const;
	mat invertNaive() const;
#ifdef OPENBLAS
	mat invertOpenBlas() const;
#endif

	mat cholesky(int &) const;
	mat choleskyOpenBlas(int &) const;
	mat choleskyNaive(int &) const;

	vec solveCholesky(const vec& b) const;
	vec solveCholeskyOpenBlas(const vec& b) const;
	vec solveCholeskyNaive(const vec& b) const;

	void addEpsilonToDiagonal(double epsilon);

	mat lu(int* pivot, int &) const;
	mat luOpenBlas(int* pivot,int &) const;
	mat luNaive(int *pivot,int &) const;

	vec solveLU(int *pivot, const vec& b) const;
	vec solveLUOpenBlas(int *pivot, const vec& b) const;
	vec solveLUNaive(int*p, const vec& b) const;


	vec solve(const vec& b) const;




	static void print_matrix(const double *a, int n);
	static double* lu_decomposition(double *a, int n, int *p, int *ret);
	static void print_permutation(int *p, int n);
	static void lu_solve(double *LU, int *p, double *b, double *x, int n);
	static void forward_substitution(int n, double *LU, int *p, double *b, double *y);
	static void backward_substitution(int n, double *LU, double *y, double *x);
	static double* matvecmul(double *a, double *b, int n);
	static void matmatmul(double *A, double *B, double *C, int n);


	mat normalizeMatrix(vec xmin, vec xmax) const;
	mat normalizeMatrix(double xmin, double xmax) const;

	mat concatenateRowWise(const mat& other) const;
	mat concatenateColumnWise(const mat& other) const;

};






} /* namespace Rodop */

#endif /* MAT_H_ */
