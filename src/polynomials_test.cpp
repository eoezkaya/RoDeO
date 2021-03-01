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

#include "polynomials_test.hpp"
#include "polynomials.hpp"
#include "auxiliary_functions.hpp"


#include<gtest/gtest.h>

TEST(testPolynomials, testPolynomialEvaluate){

	Polynomial p1(3);
	vec coeffs(4);
	coeffs(0) = 0.5; coeffs(1) = 1.5; coeffs(2) = 0.27; coeffs(3) = -1.6;
	p1.setCoefficients(coeffs);
	double x = 1.57;

	double fPolyEval = p1.evaluate(x);
	double fVal = 0.5 + 1.5 * x + 0.27 * x*x -1.6*x*x*x;
	double error = fabs(fPolyEval - fVal);
	ASSERT_LT(error,10E-10);

}

TEST(testPolynomials, testPolynomialDifferentiate){


	Polynomial p1(3);
	vec coeffs(4);
	coeffs(0) = 0.5; coeffs(1) = 1.5; coeffs(2) = 0.27; coeffs(3) = -1.6;
	p1.setCoefficients(coeffs);
	double x = 1.57;

	double fPolyDiff = p1.differentiate(x);
	double fVal = 1.5  + 2* 0.27 * x -3*1.6*x*x;
	double error = fabs(fPolyDiff - fVal);
	ASSERT_LT(error,10E-10);
}
//
//void testPolynomialProductEvaluate(void){
//	cout<<__func__<<":";
//
//	PolynomialProduct Prod;
//	Polynomial p1(3);
//
//	p1.coefficients(0) = 0.5;
//	p1.coefficients(1) = 1.5;
//	p1.coefficients(2) = 0.27;
//	p1.coefficients(3) = -1.6;
//
//	Polynomial p2(2);
//
//	p2.coefficients(0) = 0.5;
//	p2.coefficients(1) = 0.0;
//	p2.coefficients(2) = 1.0;
//
//
//
//
//	Prod.polynomials.push_back(p1);
//	Prod.polynomials.push_back(p2);
//
//	Prod.print();
//
//	//	abort();
//
//	double x1 = 1.57;
//	double x2 = -1.92;
//
//	double p1Val = 0.5 + 1.5 * x1 + 0.27 * x1*x1 -1.6*x1*x1*x1;
//	double p2Val = 0.5 + 1.0 * x2*x2 ;
//
//	rowvec x(2); x(0) = x1; x(1) = x2;
//
//	double product = Prod.evaluate(x);
//
//	bool passTest = checkValue(product,p1Val*p2Val);
//	abortIfFalse(passTest);
//
//	cout<<"\t passed\n";
//
//
//
//}
//
//void testPolynomialProductDifferentiate(void){
//	cout<<__func__<<":";
//
//	PolynomialProduct Prod;
//	Polynomial p1(3);
//
//	p1.coefficients(0) = 0.5;
//	p1.coefficients(1) = 1.5;
//	p1.coefficients(2) = 0.27;
//	p1.coefficients(3) = -1.6;
//
//	Polynomial p2(2);
//
//	p2.coefficients(0) = 0.5;
//	p2.coefficients(1) = 0.0;
//	p2.coefficients(2) = 1.0;
//
//
//
//
//	Prod.polynomials.push_back(p1);
//	Prod.polynomials.push_back(p2);
//
//	Prod.print();
//
//
//	double x1 = 1.57;
//	double x2 = -1.92;
//
//	double p1Val = 0.5 + 1.5 * x1 + 0.27 * x1*x1 -1.6*x1*x1*x1;
//	double p2Val = 0.5 + 1.0 * x2*x2 ;
//
//	rowvec x(2); x(0) = x1; x(1) = x2;
//
//	double dproddx1=Prod.differentiate(x,0);
//	double dproddx2=Prod.differentiate(x,1);
//
//	double dproddx1Exact = (1.5  + 2* 0.27 * x1 -3*1.6*x1*x1)* p2Val;
//	double dproddx2Exact = p1Val* (2.0*x2);
//
//
//	bool passTest = checkValue(dproddx1,dproddx1Exact);
//	abortIfFalse(passTest);
//	passTest = checkValue(dproddx2,dproddx2Exact);
//	abortIfFalse(passTest);
//
//	cout<<"\t passed\n";
//
//
//
//}


