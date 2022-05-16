/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "linear_solver.hpp"
#include "auxiliary_functions.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<gtest/gtest.h>



class CholeskySystemTest : public ::testing::Test {
protected:
	void SetUp() override {

		unsigned int dim = 10;
		test.setDimension(dim);
		mat M = randu<mat>(dim,dim);
		mat A = M*trans(M);


		test.setMatrix(A);

	}

	//  void TearDown() override {}

	CholeskySystem test;
};

TEST_F(CholeskySystemTest, DoesConstructorWorks) {

	EXPECT_TRUE(test.checkDimension(10) );
	EXPECT_FALSE(test.isFactorizationDone());
}



TEST_F(CholeskySystemTest, testCholeskyDecomposition){

	test.factorize();

	mat L = test.getLowerDiagonalMatrix();

	mat A = test.getMatrix();
	mat Atest = L*trans(L);

	mat error = A - Atest;
	ASSERT_TRUE(error.is_zero(10E-6));


}

TEST_F(CholeskySystemTest, testCholeskyDecompositionFails){


	mat A(10,10,fill::randu);
	test.setMatrix(A);

	test.factorize();

	ASSERT_FALSE(test.isFactorizationDone());

}




TEST_F(CholeskySystemTest, testsolveLinearSystem){



	vec x = randu<vec>(test.getDimension());

	mat A = test.getMatrix();
	vec b = A*x;

	test.setMatrix(A);
	test.factorize();

	vec xsol = test.solveLinearSystem(b);

	vec error = x-xsol;

	ASSERT_TRUE(error.is_zero(10E-6));

}

TEST_F(CholeskySystemTest, testcalculateDeterminant){


	test.factorize();

	double detTest = test.calculateDeterminant();
	mat A = test.getMatrix();


	double detCheck = det(A);
	double error = fabs(detTest-detCheck);

	ASSERT_LT(error, 10E-6);

}

TEST_F(CholeskySystemTest, testcalculateLogDeterminant){


	test.factorize();

	double detTest = test.calculateLogDeterminant();
	mat A = test.getMatrix();


	double detCheck  = log_det_sympd(A);


	double error = fabs(detTest-detCheck);

	ASSERT_LT(error, 10E-6);

}





