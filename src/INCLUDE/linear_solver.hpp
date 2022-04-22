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

#ifndef LINEARSYSTEMSOLVE_HPP
#define LINEARSYSTEMSOLVE_HPP

#include <armadillo>


using namespace arma;

class CholeskySystem{

private:

	unsigned int dimension = 0;
	mat A;
	mat L;
	mat U;
	bool ifFactorizationIsDone = false;
	bool ifMatrixIsSet = false;

	vec forwardSubstitution(vec rhs) const;
	vec backwardSubstitution(vec rhs) const;

public:


	CholeskySystem(unsigned int);
	CholeskySystem(){};

	void setDimension(unsigned int);
	unsigned int getDimension(void) const;
	bool checkDimension(unsigned int);

	mat getLowerDiagonalMatrix(void) const;
	void setMatrix(mat);
	mat getMatrix(void) const;
	void factorize();


	vec solveLinearSystem(vec) const;




};

#endif
