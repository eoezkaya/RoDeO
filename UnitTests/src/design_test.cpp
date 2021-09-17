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


#include "design.hpp"
#include "matrix_vector_operations.hpp"
#include "random_functions.hpp"
#include<gtest/gtest.h>


TEST(testDesign, testCDesignExpectedImprovementRandomGenerate){

	CDesignExpectedImprovement design(5,2);
	ASSERT_EQ(design.dim,5);
	ASSERT_EQ(design.constraintValues.size(),2);

	design.generateRandomDesignVector();

	for(unsigned int i=0; i<5; i++){

		ASSERT_LT(design.dv(i),0.2);

	}

}

TEST(testDesign, testgradientUpdateDesignVector){


	rowvec dv(3);
	dv(0) = 0.1; dv(1) = 0.2; dv(2) = 0.3;
	rowvec grad(3);
	grad(0) = 0.001; grad(1) = 2.0; grad(2) = -1.0;
	CDesignExpectedImprovement design(dv);

	design.gradientUpdateDesignVector(grad,10.0);

	ASSERT_LT(fabs(design.dv(0)-0.11),10E-10);
	ASSERT_EQ(design.dv(1), 1.0/3.0);
	ASSERT_EQ(design.dv(2), 0.0);


}

TEST(testDesign, testgenerateRandomDesignVectorAroundASample){

	unsigned int dim =  generateRandomInt(5,10);
	unsigned int N =  generateRandomInt(20,30);


	mat samples(N,dim,fill::randu);

	samples *=(1.0/dim);

	unsigned int randomIndex =  generateRandomInt(0,N-1);


	rowvec randomSample = samples.row(randomIndex);

	vec lb(dim);
	vec ub(dim);

	lb.fill(0.0);
	ub.fill(1.0/dim);

	CDesignExpectedImprovement testDesign(dim);

	testDesign.generateRandomDesignVectorAroundASample(randomSample,lb,ub);


}

TEST(testDesign, testsaveToAFile){


	rowvec dv(5);
	dv(0) = 1.0; dv(1) = -1.0; dv(2) = 2.0; dv(3) = -2.0; dv(4) = 2.0;
	Design testDesign(dv);
	testDesign.objectiveFunctionValue = 1.22;
	testDesign.tag = "Global Optimum Design";
	testDesign.ID = 22;

	testDesign.numberOfConstraints = 2;
	rowvec constraints(2);
	constraints(0) = 1.3; constraints(1) = 0.12;
	testDesign.constraintTrueValues = constraints;

	testDesign.saveToAFile("testDesign.dat");

}


