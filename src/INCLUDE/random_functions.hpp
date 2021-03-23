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

#ifndef RANDOM_FUNCTIONS_HPP
#define RANDOM_FUNCTIONS_HPP
#include <armadillo>

using namespace arma;

int generateRandomInt(int a, int b);
int generateRandomInt(uvec indices);

double generateRandomDouble(double a, double b);
void generateRandomDoubleArray(double *xp,double a, double b, unsigned int dim);


rowvec generateRandomRowVector(vec lb, vec ub);
rowvec generateRandomRowVector(double lb, double ub, unsigned int dim);

vec generateRandomVector(vec lb, vec ub);
vec generateRandomVector(double lb, double ub, unsigned int dim);

void generateRandomVector(vec lb, vec ub, unsigned int dim, double *x);

double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor);


void generateKRandomIntegers(uvec &numbers, unsigned int N, unsigned int k);
mat generateRandomWeightMatrix(unsigned int dim);




#endif
