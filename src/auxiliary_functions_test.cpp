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

#include "auxiliary_functions.hpp"
#include<gtest/gtest.h>

TEST(testAuxiliaryFunctions, testifGoogleTestWorks){


}


TEST(testAuxiliaryFunctions, testcalculatePolynomial){

	double x = 1.2;
	rowvec coeffs(3);
	coeffs(0) = 1.0;
	coeffs(1) = 2.0;
	coeffs(2) = 4.0;

	double result = calculatePolynomial(x, coeffs);
	EXPECT_EQ(result, 9.16);


}

TEST(testAuxiliaryFunctions, testsolveLinearSystemCholesky){


	unsigned int dim = 10;

	mat M(dim,dim,fill::randu);
	mat A = M*trans(M);

	mat U = chol(A);

	vec b(dim,fill::randu);
	vec x(dim,fill::randu);

	solveLinearSystemCholesky(U, x, b);

	vec Xexact = inv(A)*b;

	for(unsigned int i=0; i<dim; i++){

		double err = fabs(x(i) - Xexact(i));
		EXPECT_LE(err, 10E-6);

	}




}


TEST(testAuxiliaryFunctions, testpdf){

	double x = 1.6;
	double sigma = 1.8;
	double mu = 0.9;

	double result = pdf(x, mu, sigma);

	double err =fabs( result-	0.2054931699076307154202);

	EXPECT_LE(err, 10E-10);



}

TEST(testAuxiliaryFunctions, testcdf){

	double x = 1.6;
	double sigma = 1.8;
	double mu = 0.9;

	double result = cdf(x, mu, sigma);

	double err =fabs( result-	0.6513208290612620373879);

	EXPECT_LE(err, 10E-10);



}


