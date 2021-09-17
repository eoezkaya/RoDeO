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

#ifndef BOUNDS_HPP
#define BOUNDS_HPP

#include <armadillo>


using namespace arma;

class Bounds{

private:

	unsigned int dimension=0;
	vec lowerBounds;
	vec upperBounds;

	bool ifBoundsAreSet = false;

public:

	Bounds();
	Bounds(unsigned int);
	Bounds(vec , vec);

	unsigned int getDimension(void) const;
	vec getLowerBounds(void) const;
	double getLowerBound(unsigned int) const;
	double getUpperBound(unsigned int) const;

	vec getUpperBounds(void) const;

	bool checkIfBoundsAreValid(void) const;
	void setBounds(vec, vec);
	void setBounds(double, double);

	bool areBoundsSet(void) const;

	bool isPointWithinBounds(const vec &) const;

	void print(void) const;



};

#endif
