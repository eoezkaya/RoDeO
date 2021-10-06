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


#include "optimization.hpp"
#include "matrix_vector_operations.hpp"
#include "design.hpp"
#include<gtest/gtest.h>


#define TESTOPTIMIZATION

#ifdef TESTOPTIMIZATION

//TEST(testOptimizer, testaddPenaltyToExpectedImprovementForConstraints){
//
//	mat samples(100,3);
//
//	/* we construct first test data using the function x1*x1 + x2 * x2 */
//	for (unsigned int i=0; i<samples.n_rows; i++){
//		rowvec x(3);
//		x(0) = generateRandomDouble(0.5,1.0);
//		x(1) = generateRandomDouble(0.5,1.0);
//
//		x(2) = x(0)*x(0) + x(1)*x(1);
//		samples.row(i) = x;
//
//	}
//	samples(0,0) = 0.0; samples(0,1) = 0.0; samples(0,2) = 0.0;
//
//
//	vec lb(2); lb.fill(0.0);
//	vec ub(2); ub.fill(1.0);
//
//
//	saveMatToCVSFile(samples,"ObjFuncTest.csv");
//
//
//	/* we construct test data for the constraint x1 + x2 > 0.5*/
//	for (unsigned int i=0; i<samples.n_rows; i++){
//		rowvec x(3);
//		x(0) = generateRandomDouble(0.5,1.0);
//		x(1) = generateRandomDouble(0.5,1.0);
//
//		x(2) = x(0) + x(1);
//		samples.row(i) = x;
//
//	}
//	saveMatToCVSFile(samples,"Constraint1.csv");
//
//
//
//	ObjectiveFunction objFunc("ObjFuncTest", 2);
//	objFunc.setParameterBounds(lb,ub);
//
//
//	ConstraintFunction constraintFunc("Constraint1",2);
//	constraintFunc.setParameterBounds(lb,ub);
//
//	ConstraintDefinition def("Constraint1 > 0.5");
//
//	constraintFunc.setInequalityConstraint(def);
//
//	std::string studyName = "testOptimizer";
//	COptimizer testStudy(studyName, 2);
//	testStudy.addObjectFunction(objFunc);
//	testStudy.addConstraint(constraintFunc);
//
//	testStudy.initializeSurrogates();
//	objFunc.initializeSurrogate();
//	constraintFunc.initializeSurrogate();
//
//
//	rowvec dv(2); dv(0) = 0.01; dv(1) = 0.01;
//	CDesignExpectedImprovement testDesign(dv,1);
//	objFunc.calculateExpectedImprovement(testDesign);
//
//
//	testStudy.addPenaltyToExpectedImprovementForConstraints(testDesign);
//
//	EXPECT_EQ(testDesign.valueExpectedImprovement, 0.0);
//
//}


//TEST(testOptimizer, testaddPenaltyToExpectedImprovementForConstraintsWithTwoConstraints){
//
//	/* in this test, we have two constraints. One is satisfied the other not */
//	mat samples(100,3);
//
//	/* we construct first test data using the function x1*x1 + x2 * x2 */
//	for (unsigned int i=0; i<samples.n_rows; i++){
//		rowvec x(3);
//		x(0) = generateRandomDouble(0.5,1.0);
//		x(1) = generateRandomDouble(0.5,1.0);
//
//		x(2) = x(0)*x(0) + x(1)*x(1);
//		samples.row(i) = x;
//
//	}
//	samples(0,0) = 0.0; samples(0,1) = 0.0; samples(0,2) = 0.0;
//
//
//	vec lb(2); lb.fill(0.0);
//	vec ub(2); ub.fill(1.0);
//
//
//	saveMatToCVSFile(samples,"ObjFuncTest.csv");
//
//
//	/* we construct test data for the first constraint x1 + x2 > 0.5*/
//	for (unsigned int i=0; i<samples.n_rows; i++){
//		rowvec x(3);
//		x(0) = generateRandomDouble(0.5,1.0);
//		x(1) = generateRandomDouble(0.5,1.0);
//
//		x(2) = x(0) + x(1);
//		samples.row(i) = x;
//
//	}
//	saveMatToCVSFile(samples,"Constraint1.csv");
//
//
//	/* we construct test data for the second constraint x1 < 0.1*/
//	for (unsigned int i=0; i<samples.n_rows; i++){
//		rowvec x(3);
//		x(0) = generateRandomDouble(0.5,1.0);
//		x(1) = generateRandomDouble(0.5,1.0);
//
//		x(2) = x(0);
//		samples.row(i) = x;
//
//	}
//	saveMatToCVSFile(samples,"Constraint2.csv");
//
//
//
//
//	ObjectiveFunction objFunc("ObjFuncTest", 2);
//	objFunc.setParameterBounds(lb,ub);
//
//
//	ConstraintFunction constraintFunc("Constraint1",2);
//	constraintFunc.setParameterBounds(lb,ub);
//
//	ConstraintDefinition def("Constraint1 > 0.5");
//	constraintFunc.setInequalityConstraint(def);
//
//
//	ConstraintDefinition def2("Constraint2 < 0.1");
//	ConstraintFunction constraintFunc2("Constraint2",2);
//	constraintFunc2.setParameterBounds(lb,ub);
//
//	constraintFunc2.setInequalityConstraint(def2);
//
//
//	std::string studyName = "testOptimizer";
//	COptimizer testStudy(studyName, 2);
//	testStudy.addObjectFunction(objFunc);
//	testStudy.addConstraint(constraintFunc);
//	testStudy.addConstraint(constraintFunc2);
//
//	testStudy.initializeSurrogates();
//	objFunc.initializeSurrogate();
//
//
//	rowvec dv(2); dv(0) = 0.01; dv(1) = 0.01;
//	CDesignExpectedImprovement testDesign(dv,2);
//	objFunc.calculateExpectedImprovement(testDesign);
//
//	testStudy.addPenaltyToExpectedImprovementForConstraints(testDesign);
//
//	EXPECT_EQ(testDesign.valueExpectedImprovement, 0.0);
//
//}


TEST(testOptimizer, testfindTheMostPromisingDesign){

	mat samples(100,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(0.5,1.0);
		x(1) = generateRandomDouble(0.5,1.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}
	samples(0,0) = 0.0; samples(0,1) = 0.0; samples(0,2) = 0.0;


	vec lb(2); lb.fill(0.0);
	vec ub(2); ub.fill(1.0);


	saveMatToCVSFile(samples,"ObjFuncTest.csv");

	ObjectiveFunction objFunc("ObjFuncTest", 2);
	objFunc.setParameterBounds(lb,ub);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	objFunc.setParametersByDefinition(testObjectiveFunctionDef);


	std::string studyName = "testOptimizer";
	Optimizer testStudy(studyName, 2);
	testStudy.addObjectFunction(objFunc);
	testStudy.initializeSurrogates();

	testStudy.findTheMostPromisingDesign();

	CDesignExpectedImprovement testDesign = testStudy.getDesignWithMaxExpectedImprovement();


	EXPECT_LT(testDesign.dv(0), 0.1);
	EXPECT_LT(testDesign.dv(1), 0.1);

}


TEST(testOptimizer, testMaximizeEIGradientBased){

	mat samples(10,3);


	 samples(0,0) = 1.4199;  samples(0,1) = 0.2867; samples(0,2) = 2.0982;
	 samples(1,0) = 1.2761;  samples(1,1) = 0.7363; samples(1,2) = 2.1706;
	 samples(2,0) =-4.8526;  samples(2,1) =-1.3832; samples(2,2) = 25.4615;
	 samples(3,0) =-4.9643;  samples(3,1) = 2.5681; samples(3,2) = 31.2395;
	 samples(4,0) =-3.2966;  samples(4,1) = 3.7140; samples(4,2) = 24.6612;
	 samples(5,0) = 2.1202;  samples(5,1) = 1.5686; samples(5,2) =  6.9559;
	 samples(6,0) = 4.0689;  samples(6,1) = 3.8698; samples(6,2) = 31.5311;
	 samples(7,0) = 3.6139;  samples(7,1) =-0.5469; samples(7,2) = 13.3596;
	 samples(8,0) = 4.5835;  samples(8,1) =-3.9091; samples(8,2) = 36.2896;
	 samples(9,0) = 3.8641;  samples(9,1) = 2.3025; samples(9,2) = 20.2327;

	vec lb(2); lb.fill(-5.0);
	vec ub(2); ub.fill(5.0);


	saveMatToCVSFile(samples,"ObjFuncTest.csv");

	ObjectiveFunction objFunc("ObjFuncTest", 2);
	objFunc.setParameterBounds(lb,ub);

	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
	objFunc.setParametersByDefinition(testObjectiveFunctionDef);

	objFunc.initializeSurrogate();


	std::string studyName = "testOptimizer";
	Optimizer testStudy(studyName, 2);
	testStudy.addObjectFunction(objFunc);
	testStudy.initializeSurrogates();

	rowvec dv(2); dv(0) = 0.29; dv(1) = 0.29;

	CDesignExpectedImprovement initialDesign(dv);


	CDesignExpectedImprovement optimizedDesign = testStudy.MaximizeEIGradientBased(initialDesign);

	ASSERT_GT(optimizedDesign.valueExpectedImprovement, initialDesign.valueExpectedImprovement + 1.0);


}
#endif

