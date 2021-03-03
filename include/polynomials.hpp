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

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP
#include <vector>
#include <armadillo>
using namespace arma;


class Polynomial{

private:

	unsigned int degree;
	vec coefficients;

public:

	Polynomial();
	Polynomial(unsigned int);

	double evaluate(double x) const;
	double differentiate(double x) const;
	void print(void) const;
	void initializeRandomCoefficients(void);
	void setCoefficients(vec);


};

class PolynomialProduct{

private:

	std::vector<Polynomial> polynomials;

public:

	PolynomialProduct();
	PolynomialProduct(unsigned int, unsigned int);

	double evaluate(const rowvec &x) const;
	double differentiate(const rowvec &x, unsigned int indx) const;
	void print(void) const;
	void initializeRandomCoefficients(void);

	friend void testPolynomialProductEvaluate(void);
	friend void testPolynomialProductDifferentiate(void);

};

void generateRandomTestAndValidationDataForGradientModels(int dim, int N);


#endif
