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
#ifndef LINEARSYSTEMSOLVE_HPP
#define LINEARSYSTEMSOLVE_HPP

#include "matrix.hpp"
#include "vector.hpp"
#include<vector>
#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif

using namespace std;

namespace Rodop {

class CholeskySystem{

private:

	unsigned int dimension = 0;
	mat A;
	mat factorizationMatrix;
	bool ifFactorizationIsDone = false;
	bool ifMatrixIsSet = false;


public:

	bool ifDisplay = false;

	CholeskySystem(unsigned int);
	CholeskySystem(){};

	void setDimension(unsigned int);
	unsigned getDimension(void) const;
	bool checkDimension(unsigned int);

	mat getFactorizedMatrix(void) const;
	void setMatrix(mat);
	mat getMatrix(void) const;
	void factorize();
	bool isFactorizationDone(void);

	double calculateDeterminant(void);
	double calculateLogDeterminant(void);

	vec solveLinearSystem(const vec &b) const;

};

class LUSystem{

private:

	unsigned int dimension = 0;
	mat A;
	mat factorizationMatrix;
	vector<int> pivots;
	bool ifFactorizationIsDone = false;
	bool ifMatrixIsSet = false;


public:

	LUSystem(unsigned int);
	LUSystem(){};

	void setDimension(unsigned int);
	int getDimension(void) const;
	bool checkDimension(unsigned int);

	mat getFactorizedMatrix(void) const;
	void setMatrix(mat);
	mat getMatrix(void) const;
	void factorize();
	bool isFactorizationDone(void);

	vec solveLinearSystem(const vec &b);

};


}

#endif
