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

#include "../../../LinearAlgebra/INCLUDE/linear_solver.hpp"
#include "../../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../INCLUDE/test_defines.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<gtest/gtest.h>

#ifdef LINEAR_SOLVER_TEST

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


class SVDSystemTest : public ::testing::Test {
protected:
	void SetUp() override {

		A = generateRandomCorrelationMatrix(50);
		test.setMatrix(A);


	}

	mat generateRandomCorrelationMatrix(unsigned int dim){

		vec X(dim,fill::randu);
		mat R(dim,dim, fill::zeros);
		for(unsigned int i=0; i<dim; i++){

			for(unsigned int j=0; j<dim; j++){

				double dist  = (X[i] - X[j])*(X[i] - X[j]);
				R(i,j) = exp(-dist);

			}

		}

		return R;

	}


	//  void TearDown() override {}

	SVDSystem test;
	mat A;
};

TEST_F(SVDSystemTest, DoesConstructorWorks) {

	EXPECT_FALSE(test.ifFactorizationIsDone);
	EXPECT_TRUE(test.ifMatrixIsSet);

}


TEST_F(SVDSystemTest, testSVDfactorize){

	test.factorize();
	EXPECT_TRUE(test.ifFactorizationIsDone);

}

TEST_F(SVDSystemTest, testSVDsolveLinearSystem){

	test.factorize();
	vec xExact(50,fill::randu);
	vec rhs = A*xExact;

	vec x = test.solveLinearSystem(rhs);

	vec r = A*x-rhs;

//	r.print("res");
	double normResidual = norm(r,2);
//	printScalar(normResidual);


	EXPECT_LT(normResidual,10E-8);


}

TEST_F(SVDSystemTest, testSVDsolveLinearSystemRectangularMatrix){

	unsigned int m = 50;
	unsigned int n = 30;
	mat A(m, n, fill::randu);

	test.setMatrix(A);
	test.factorize();
	vec xExact(n,fill::randu);
	vec rhs = A*xExact;

	vec x = test.solveLinearSystem(rhs);

	vec r = A*x-rhs;

	double normResidual = norm(r,2);
//	printScalar(normResidual);

	EXPECT_LT(normResidual,10E-8);


}

TEST_F(SVDSystemTest, testcalculateLogDeterminant){

	unsigned int n = 50;
	mat M(n, n, fill::randu);
	mat MTM = trans(M)*M;
	test.setMatrix(MTM);
	test.factorize();

	double detTest = test.calculateLogAbsDeterminant();

	double detCheck  = log_det_sympd(MTM);
	double error = fabs(detTest-detCheck);

	ASSERT_LT(error, 10E-6);
}

#endif
