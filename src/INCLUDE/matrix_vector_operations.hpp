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
#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include "bounds.hpp"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace arma;

void abortIfHasNan(rowvec &);

bool isEqual(const mat &A, const mat&B, double tolerance);

void printMatrix(mat M, std::string name="None");
void printVector(vec v, std::string name="None");
void printVector(rowvec v, std::string name="None");
void printVector(std::vector<std::string> v);
void printVector(std::vector<int> v);
void printVector(std::vector<bool> v);

vec normalizeColumnVector(vec x, double xmin, double xmax);

rowvec normalizeRowVector(rowvec x, vec xmin, vec xmax);
rowvec normalizeRowVectorBack(rowvec xnorm, vec xmin, vec xmax);

void copyRowVector(rowvec &a,rowvec b);
void copyRowVector(rowvec &a,rowvec b, unsigned int);

void appendRowVectorToCSVData(rowvec v, std::string fileName);

mat readMatFromCVSFile(std::string fileName);
void saveMatToCVSFile(mat M, std::string fileName);

mat normalizeMatrix(mat matrixIn);
mat normalizeMatrix(mat matrixIn, vec xmin, vec xmax);
mat normalizeMatrix(mat matrixIn, Bounds &boxConstraints);


int findInterval(double value, vec discreteValues);



#endif
