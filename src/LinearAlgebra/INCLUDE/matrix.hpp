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


#ifndef MAT_H
#define MAT_H

#include "vector.hpp"

namespace rodeo{



class mat {
private:
	int nrows;
	int ncols;
	double** matrix;

	void deallocateMemory(void);
	void allocateMemory();

public:
	// Constructor
	mat(int defaultRows = 0, int defaultCols = 0);
	mat(int r, int c, double val);

	// Copy constructor
	mat(const mat& other);

	// Destructor
	~mat();

	int getNRows(void) const;
	int getNCols(void) const;
	int getSize(void) const;


	// Matrix addition
	mat operator+(const mat& other) const;

	// Matrix subtraction
	mat operator-(const mat& other) const;

	// Matrix multiplication
	mat operator*(const mat& other) const;
	mat operator*(const double scalar) const;

	double& operator()(int i, int j);

	// Const version of the access operator for read-only access
	const double& operator()(int i, int j) const;

	// Display matrix
	void print(void) const;
	void display(void) const{
		print();
	}

	void fill(double value);
	void fillRandom(void);

	static mat eye(int size) {
		mat identity(size, size);

		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				if (i == j) {
					identity(i, j) = 1.0;
				} else {
					identity(i, j) = 0.0;
				}
			}
		}

		return identity;
	}

	void resize(int newRows, int newCols);
	mat submat(int startRow, int startCol, int subRows, int subCols) const;
	void addColumns(int numColumnsToAdd, double defaultValue = 0.0);
	void addRows(int numRowsToAdd, double defaultValue = 0.0);

	mat concatenateRowWise(const mat& other) const;
	mat concatenateColumnWise(const mat& other) const;

//	vec getRow(int n) const;

};

}

#endif // MAT_H

