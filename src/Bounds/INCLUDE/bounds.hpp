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

	void reset(void);


	void setDimension(unsigned int);
	void setBounds(vec, vec);
	void setBounds(double, double);


	bool checkIfBoundsAreValid(void) const;
	bool areBoundsSet(void) const;
	bool isPointWithinBounds(const vec &) const;

	vec generateVectorWithinBounds(void) const;

	void print(void) const;

	unsigned int getDimension(void) const;
	vec getLowerBounds(void) const;
	vec getUpperBounds(void) const;

};

#endif
