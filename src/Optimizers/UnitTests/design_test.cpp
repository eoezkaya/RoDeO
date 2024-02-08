/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), Rheinland-Pfälzische Technische Universität
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


#include "../INCLUDE/design.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../../Random/INCLUDE/random_functions.hpp"

#include<gtest/gtest.h>



class DesignTest: public ::testing::Test {
protected:
	void SetUp() override {
		testDesign.setDimension(2);
		rowvec dv(2);
		dv(0) = 1.2; dv(1) = -1.777;
		testDesign.designParameters = dv;

	}

	void TearDown() override {

	}


	Design testDesign;

};

TEST_F(DesignTest, constructor){

	Design testObject;

	ASSERT_TRUE(testObject.dimension == 0);
	ASSERT_TRUE(testObject.isDesignFeasible);
	ASSERT_EQ(testObject.ID,0);

}


TEST_F(DesignTest, constructorWithRowVector){

	rowvec dv(2);
	dv(0) = 1.2; dv(1) = 2.3;
	Design testObject(dv);

	ASSERT_TRUE(testObject.dimension == 2);
	ASSERT_EQ(testObject.designParameters.size(), 2);
	ASSERT_EQ(testObject.designParameters(0), 1.2);
	ASSERT_EQ(testObject.designParameters(1), 2.3);
	ASSERT_EQ(testObject.gradient.size(),2);
	ASSERT_EQ(testObject.gradientLowFidelity.size(),2);

}


TEST_F(DesignTest, constructorWithInt){

	Design testObject(2);

	ASSERT_TRUE(testObject.dimension == 2);
	ASSERT_EQ(testObject.designParameters.size(), 2);
	ASSERT_EQ(testObject.gradient.size(),2);
	ASSERT_EQ(testObject.gradientLowFidelity.size(),2);

}

TEST_F(DesignTest, generateRandomDifferentiationDirection){

	Design testObject(2);
	testObject.generateRandomDifferentiationDirection();
	rowvec d = testObject.tangentDirection;

	double normd = norm(d,2);
	double error = fabs(normd - 1.0);

	ASSERT_LT(error,10E-10);
	ASSERT_TRUE(d(0) > -1.0 && d(0) < 1.0);
	ASSERT_TRUE(d(1) > -1.0 && d(1) < 1.0);
}






TEST_F(DesignTest, constructSampleObjectiveFunction){

	testDesign.trueValue = 8.9;

	rowvec samples = testDesign.constructSampleObjectiveFunction();

	ASSERT_EQ(samples.size(), 3);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);

}

TEST_F(DesignTest, constructSampleObjectiveFunctionLowFi){

	testDesign.trueValueLowFidelity = 8.9;
	rowvec samples = testDesign.constructSampleObjectiveFunctionLowFi();

	ASSERT_EQ(samples.size(), 3);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);



}


TEST_F(DesignTest, constructSampleObjectiveFunctionTangent){

	rowvec direction(2);
	direction(0) = 1.0;
	direction(1) = 0.0;
	testDesign.tangentDirection = direction;
	testDesign.trueValue = 8.9;
	testDesign.tangentValue = 12.8;

	rowvec samples = testDesign.constructSampleObjectiveFunctionWithTangent();

	ASSERT_EQ(samples.size(), 6);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);
	ASSERT_EQ(samples(3), 12.8);
	ASSERT_EQ(samples(4), 1.0);
	ASSERT_EQ(samples(5), 0.0);

}

TEST_F(DesignTest, constructSampleObjectiveFunctionTangentLowFi){

	rowvec direction(2);
	direction(0) = 1.0;
	direction(1) = 0.0;
	testDesign.tangentDirection = direction;
	testDesign.trueValueLowFidelity = 8.9;
	testDesign.tangentValueLowFidelity = 12.8;

	rowvec samples = testDesign.constructSampleObjectiveFunctionWithTangentLowFi();

	ASSERT_EQ(samples.size(), 6);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);
	ASSERT_EQ(samples(3), 12.8);
	ASSERT_EQ(samples(4), 1.0);
	ASSERT_EQ(samples(5), 0.0);

}



TEST_F(DesignTest, constructSampleObjectiveFunctionGradient){

	rowvec gradient(2);
	gradient(0) = 11.0;
	gradient(1) = 6.0;
	testDesign.trueValue = 8.9;
	testDesign.gradient = gradient;
	rowvec samples = testDesign.constructSampleObjectiveFunctionWithGradient();

	ASSERT_EQ(samples.size(), 5);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);
	ASSERT_EQ(samples(3), 11.0);
	ASSERT_EQ(samples(4), 6.0);

}

TEST_F(DesignTest, constructSampleObjectiveFunctionGradientLowFi){

	rowvec gradient(2);
	gradient(0) = 11.0;
	gradient(1) = 6.0;
	testDesign.trueValueLowFidelity = 8.9;
	testDesign.gradientLowFidelity = gradient;
	rowvec samples = testDesign.constructSampleObjectiveFunctionWithGradientLowFi();

	ASSERT_EQ(samples.size(), 5);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 8.9);
	ASSERT_EQ(samples(3), 11.0);
	ASSERT_EQ(samples(4), 6.0);

}

TEST_F(DesignTest, constructSampleConstraint){


	testDesign.setNumberOfConstraints(2);
	testDesign.constraintTrueValues(0) =  2.2;
	testDesign.constraintTrueValues(1) = -5.9;

	rowvec samples = testDesign.constructSampleConstraint(0);

	ASSERT_EQ(samples.size(), 3);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);

	samples = testDesign.constructSampleConstraint(1);

	ASSERT_EQ(samples(2), -5.9);


}

TEST_F(DesignTest, constructSampleConstraintLowFi){

	testDesign.setNumberOfConstraints(1);
	testDesign.constraintTrueValuesLowFidelity(0) =  2.2;

	rowvec samples = testDesign.constructSampleConstraintLowFi(0);

	ASSERT_EQ(samples.size(), 3);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);


}



TEST_F(DesignTest, constructSampleConstraintTangent){

	testDesign.setNumberOfConstraints(1);
	testDesign.constraintTrueValues(0) =  2.2;
	testDesign.constraintTangent(0) =  4.42;
	rowvec direction1(2);
	direction1(0) = 1.0;
	direction1(1) = 0.0;
	testDesign.constraintDifferentiationDirectionsMatrix.row(0) = direction1;


	rowvec samples = testDesign.constructSampleConstraintWithTangent(0);

	ASSERT_EQ(samples.size(), 6);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);
	ASSERT_EQ(samples(3), 4.42);
	ASSERT_EQ(samples(4), 1.0);
	ASSERT_EQ(samples(5), 0.0);


}

TEST_F(DesignTest, constructSampleConstraintTangentLowFi){

	testDesign.setNumberOfConstraints(1);
	testDesign.constraintTrueValuesLowFidelity(0) =  2.2;
	testDesign.constraintTangentLowFidelity(0) =  4.42;
	rowvec direction1(2);
	direction1(0) = 1.0;
	direction1(1) = 0.0;
	testDesign.constraintDifferentiationDirectionsMatrixLowFi.row(0) = direction1;

	rowvec samples = testDesign.constructSampleConstraintWithTangentLowFi(0);

	ASSERT_EQ(samples.size(), 6);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);
	ASSERT_EQ(samples(3), 4.42);
	ASSERT_EQ(samples(4), 1.0);
	ASSERT_EQ(samples(5), 0.0);


}

TEST_F(DesignTest, constructSampleConstraintAdjoint){

	testDesign.setNumberOfConstraints(1);
	testDesign.constraintTrueValues(0) =  2.2;

	rowvec gradient(2);
	gradient(0) = 51.2;
	gradient(1) = -20.0;
	testDesign.constraintGradientsMatrix.row(0) = gradient;

	rowvec samples = testDesign.constructSampleConstraintWithGradient(0);

	ASSERT_EQ(samples.size(), 5);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);
	ASSERT_EQ(samples(3), 51.2);
	ASSERT_EQ(samples(4), -20.0);

}

TEST_F(DesignTest, constructSampleConstraintAdjointLowFi){

	testDesign.setNumberOfConstraints(1);
	testDesign.constraintTrueValuesLowFidelity(0) =  2.2;

	rowvec gradient(2);
	gradient(0) = 51.2;
	gradient(1) = -20.0;
	testDesign.constraintGradientsMatrixLowFi.row(0) = gradient;

	rowvec samples = testDesign.constructSampleConstraintWithGradientLowFi(0);

	ASSERT_EQ(samples.size(), 5);
	ASSERT_EQ(samples(0), 1.2);
	ASSERT_EQ(samples(1),-1.777);
	ASSERT_EQ(samples(2), 2.2);
	ASSERT_EQ(samples(3), 51.2);
	ASSERT_EQ(samples(4), -20.0);

}

TEST_F(DesignTest, saveToAFile){

	testDesign.trueValue = 1.22;
	testDesign.tag = "Global Optimum Design";
	testDesign.ID = 22;

	testDesign.numberOfConstraints = 2;
	rowvec constraints(2);
	constraints(0) = 1.3; constraints(1) = 0.12;
	testDesign.constraintTrueValues = constraints;

	testDesign.saveToAFile("testDesign.dat");


	remove("testDesign.dat");

}

TEST_F(DesignTest, saveToAnXMLFile){

	testDesign.trueValue = 1.22;
	testDesign.tag = "Global Optimum Design";
	testDesign.ID = 22;
	testDesign.numberOfConstraints = 2;
	rowvec constraints(2);
	constraints(0) = 1.3; constraints(1) = 0.12;
	testDesign.constraintTrueValues = constraints;

	testDesign.saveToXMLFile("testDesign.xml");

	remove("testDesign.xml");

}

TEST_F(DesignTest, readFromXmlFile){

	testDesign.trueValue = 1.22;
	testDesign.tag = "Global Optimum Design";
	testDesign.ID = 22;
	testDesign.numberOfConstraints = 2;
	rowvec constraints(2);
	constraints(0) = 1.3; constraints(1) = 0.12;
	testDesign.constraintTrueValues = constraints;

	testDesign.saveToXMLFile("testDesign.xml");

	testDesign.reset();
	testDesign.readFromXmlFile("testDesign.xml");

	testDesign.print();

	ASSERT_EQ(testDesign.ID, 22);


	remove("testDesign.xml");

}



class DesignForBayesianOptimizationTest: public ::testing::Test {
protected:
	void SetUp() override {
		testDesign.dim = 5;
		rowvec constraints(2,fill::zeros);
		rowvec dv(5,fill::zeros);
		testDesign.constraintValues = constraints;
		testDesign.dv = dv;
		testDesign.objectiveFunctionValue = 18.8;
		testDesign.sigma = 5.6;


	}

	void TearDown() override {

	}


	DesignForBayesianOptimization testDesign;

};

TEST_F(DesignForBayesianOptimizationTest, constructor){


	ASSERT_EQ(testDesign.dim,5);
	ASSERT_EQ(testDesign.constraintValues.size(),2);

}


TEST_F(DesignForBayesianOptimizationTest, generateRandomDesignVector){

	testDesign.generateRandomDesignVector();

	for(unsigned int i=0; i<5; i++){
		ASSERT_LT(testDesign.dv(i),0.2);
	}
}

TEST_F(DesignForBayesianOptimizationTest, gradientUpdateDesignVector){

	vec lb(5);
	vec ub(5);

	lb.fill(0.0);
	ub.fill(0.2);

	rowvec dv(5);
	dv(0) = 0.1; dv(1) = 0.2; dv(2) = 0.3; dv(3) = 0.3; dv(4) = 0.3;
	rowvec grad(5);
	grad(0) = 0.001; grad(1) = 2.0; grad(2) = -1.0; grad(3) = -1.0; grad(4) = -1.0;
	testDesign.dv = dv;
	testDesign.gradientUpdateDesignVector(grad,lb,ub,10.0);
	ASSERT_LT(fabs(testDesign.dv(0)- 0.11),10E-10);
	ASSERT_EQ(testDesign.dv(1), 1.0/5.0);
	ASSERT_EQ(testDesign.dv(2), 0.0);
}

TEST_F(DesignForBayesianOptimizationTest, generateRandomDesignVectorAroundASample){

	unsigned int dim =  5;
	unsigned int N =  generateRandomInt(20,30);

	mat samples(N,dim,fill::randu);
	samples *=(1.0/dim);
	unsigned int randomIndex =  generateRandomInt(0,N-1);

	rowvec randomSample = samples.row(randomIndex);
	vec lb(dim);
	vec ub(dim);
	lb.fill(0.0);
	ub.fill(1.0/dim);

	testDesign.generateRandomDesignVectorAroundASample(randomSample,lb,ub);

	double normL2 = norm(randomSample-testDesign.dv);

	EXPECT_LT(normL2, 1.0);
	EXPECT_GT(normL2, 0.0);

}


TEST_F(DesignForBayesianOptimizationTest, calculateProbalityThatTheEstimateIsGreaterThanAValue){

	double p = testDesign.calculateProbalityThatTheEstimateIsGreaterThanAValue(20.0);
	EXPECT_GT(p, 0.4);

	p = testDesign.calculateProbalityThatTheEstimateIsGreaterThanAValue(200.0);
	EXPECT_LT(p, 0.0001);

}

TEST_F(DesignForBayesianOptimizationTest, calculateProbalityThatTheEstimateIsLessThanAValue){

	double p1 = testDesign.calculateProbalityThatTheEstimateIsLessThanAValue(20.0);
	double p2 = testDesign.calculateProbalityThatTheEstimateIsGreaterThanAValue(20.0);

	EXPECT_EQ(p1+p2, 1.0);
}




