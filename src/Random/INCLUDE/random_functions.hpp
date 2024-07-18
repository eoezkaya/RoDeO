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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#ifndef RANDOM_FUNCTIONS_HPP
#define RANDOM_FUNCTIONS_HPP
#include <armadillo>

using namespace arma;

int generateRandomInt(int a, int b);
int generateRandomInt(uvec indices);

double generateRandomDouble(double a, double b);

mat generateRandomMatrix(unsigned int Nrows, unsigned int Ncols, double *lb, double*ub);
mat generateRandomMatrix(unsigned int Nrows, vec lb, vec ub);

mat generateLatinHypercubeMatrix(int N, int dimensions);
mat generateLatinHypercubeMatrix(int N, int dimensions, const vec& lowerBounds, const vec& upperBounds);

double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor);



template <typename T>
T generateRandomVector(const vec & lb, const vec & ub){
	unsigned int dim = lb.size();
	T x(dim);
	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb(i), ub(i));
	}
	return x;

}

template <typename T>
T generateRandomVector(double lb, double ub, unsigned int dim){

	T x(dim);
	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb, ub);
	}
	return x;

}


#endif
