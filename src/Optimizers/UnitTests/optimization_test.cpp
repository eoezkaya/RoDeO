/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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


#include "../INCLUDE/optimization.hpp"
#include "../../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../../TestFunctions/INCLUDE/standard_test_functions.hpp"
#include "../INCLUDE/design.hpp"

#include<gtest/gtest.h>




class OptimizationTest : public ::testing::Test {
protected:
	void SetUp() override {


		objFunHimmelblau.setDimension(2);

		lb = zeros<vec>(2);
		ub = zeros<vec>(2);
		lb.fill(-6.0);
		ub.fill(6.0);

		boxConstraints.setBounds(lb, ub );


		definition.designVectorFilename = "dv.dat";
		definition.executableName = "himmelblau";
		definition.executableNameLowFi = "himmelblauLowFi";
		definition.outputFilename = "objFunVal.dat";
		definition.outputFilenameLowFi = "objFunVal.dat";
		definition.name= "himmelblau";
		definition.nameHighFidelityTrainingData = "himmelblau.csv";
		definition.nameLowFidelityTrainingData = "himmelblauLowFi.csv";
		definition.ifDefined = true;


		constraintFunc1.setDimension(2);
		constraintFunc2.setDimension(2);


		constraintDefinition1.designVectorFilename = "dv.dat";
		constraintDefinition1.executableName = "constraint1";
		constraintDefinition1.outputFilename = "constraintFunction1.dat";
		constraintDefinition1.name = "constraint1";
		constraintDefinition1.nameHighFidelityTrainingData = "constraint1.csv";
		constraintDefinition1.ifDefined = true;

		std::string defitinition1 = "constraint1 < 10.0";

		constraintDef1.setDefinition(defitinition1);




		constraintDefinition2.name = "constraint2";
		constraintDefinition2.designVectorFilename = "dv.dat";
		constraintDefinition2.executableName = "constraint2";
		constraintDefinition2.outputFilename = "constraintFunction2.dat";
		constraintDefinition2.nameHighFidelityTrainingData = "constraint2.csv";
		constraintDefinition2.ifDefined = true;

		std::string defitinition2 = "constraint2 > 3.0";
		constraintDef2.setDefinition(defitinition2);



		testOptimizer.setDimension(2);
		testOptimizer.setName("HimmelblauOptimization");
		//		testOptimizer.setMaximumNumberOfIterationsForEIMaximization(1000000);


		himmelblauFunction.function.filenameTrainingData = "himmelblau.csv";
		himmelblauFunction.function.filenameTrainingDataHighFidelity = "himmelblau.csv";
		himmelblauFunction.function.filenameTrainingDataLowFidelity = "himmelblauLowFi.csv";
		himmelblauFunction.function.numberOfTrainingSamples = 50;
		himmelblauFunction.function.numberOfTrainingSamplesLowFi = 100;

		himmelblauFunction.function.ifInputSamplesAreGenerated = false;

		constraint1.function.filenameTrainingData = "constraint1.csv";
		constraint1.function.numberOfTrainingSamples = 40;
		constraint1.function.ifInputSamplesAreGenerated = false;

		constraint2.function.filenameTrainingData = "constraint2.csv";
		constraint2.function.numberOfTrainingSamples = 40;
		constraint1.function.ifInputSamplesAreGenerated = false;




	}


	void prepareObjectiveFunction(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblau.cpp", definition.executableName);
		himmelblauFunction.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}


	void prepareObjectiveFunctionWithAdjoint(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblauAdjoint.cpp", definition.executableName);

		himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
		definition.modelHiFi = GRADIENT_ENHANCED;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}




	void prepareObjectiveFunctionWithTangent(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblauTangent.cpp", definition.executableName);

		himmelblauFunction.function.generateTrainingSamplesWithTangents();
		definition.modelHiFi = TANGENT_ENHANCED;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}

	void prepareObjectiveFunctionWithML(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblau.cpp", definition.executableName);
		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblauLowFidelity.cpp", definition.executableNameLowFi);
		himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

		definition.ifMultiLevel = true;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);
	}

	void prepareObjectiveFunctionWithMLLowFiAdjoint(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblau.cpp", definition.executableName);
		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/himmelblauAdjointLowFi.cpp", definition.executableNameLowFi);
		himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();

		definition.ifMultiLevel = true;
		definition.modelHiFi = ORDINARY_KRIGING;
		definition.modelLowFi = GRADIENT_ENHANCED;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);
	}



	void prepareFirstConstraint(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/constraint1.cpp", constraintDefinition1.executableName);

		mat X = himmelblauFunction.function.trainingSamplesInput;
		X = shuffleRows(X);
		mat Xreduced = X.submat(0, 0, constraint1.function.numberOfTrainingSamples -1, 1);
		constraint1.function.trainingSamplesInput = Xreduced;
		constraint1.function.ifInputSamplesAreGenerated = true;
		constraint1.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);

		constraintFunc1.setParametersByDefinition(constraintDefinition1);
		constraintFunc1.setConstraintDefinition(constraintDef1);
		constraintFunc1.setID(0);
		testOptimizer.addConstraint(constraintFunc1);

	}
	void prepareSecondConstraint(void){

		compileWithCpp("../../../src/Optimizers/UnitTests/Auxiliary/constraint2.cpp", constraintDefinition2.executableName);
		mat X = himmelblauFunction.function.trainingSamplesInput;
		X = shuffleRows(X);
		mat Xreduced = X.submat(0, 0, constraint2.function.numberOfTrainingSamples -1, 1);
		constraint2.function.trainingSamplesInput = Xreduced;
		constraint2.function.ifInputSamplesAreGenerated = true;
		constraint2.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);
		constraintFunc2.setParametersByDefinition(constraintDefinition2);
		constraintFunc2.setConstraintDefinition(constraintDef2);
		constraintFunc2.setID(1);
		testOptimizer.addConstraint(constraintFunc2);

	}

	void TearDown() override {



	}

	std::string problemName = "himmelblauOptimization";
	Optimizer testOptimizer;
	ObjectiveFunction objFunHimmelblau;
	ConstraintDefinition constraintDef1;
	ConstraintDefinition constraintDef2;
	ConstraintFunction constraintFunc1;
	ConstraintFunction constraintFunc2;

	HimmelblauFunction himmelblauFunction;
	HimmelblauConstraintFunction1 constraint1;
	HimmelblauConstraintFunction2 constraint2;

	ObjectiveFunctionDefinition definition;
	ObjectiveFunctionDefinition constraintDefinition1;
	ObjectiveFunctionDefinition constraintDefinition2;

	vec lb;
	vec ub;

	Bounds boxConstraints;


};


TEST_F(OptimizationTest, constructor){

	ASSERT_FALSE(testOptimizer.ifBoxConstraintsSet);
	ASSERT_FALSE(testOptimizer.ifObjectFunctionIsSpecied);
	ASSERT_FALSE(testOptimizer.isConstrained());
}

TEST_F(OptimizationTest, setBoxConstraints){

	prepareObjectiveFunction();
	testOptimizer.setBoxConstraints(boxConstraints);

	ASSERT_TRUE(testOptimizer.ifBoxConstraintsSet);

}

TEST_F(OptimizationTest, trimVectorSoThatItStaysWithinTheBounds){

	testOptimizer.dimension = 2;

	rowvec x(2);
	x(0) = -1;
	x(1) = 0.6;

	testOptimizer.trimVectorSoThatItStaysWithinTheBounds(x);

	double err = fabs(x(0) - 0.0);
	ASSERT_LT(err, 10E-10);
	err = fabs(x(1) - 0.5);
	ASSERT_LT(err, 10E-10);

}

TEST_F(OptimizationTest, setOptimizationHistory){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();

	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();

	testOptimizer.setOptimizationHistory();

	unsigned int N = himmelblauFunction.function.numberOfTrainingSamples;
	ASSERT_EQ(testOptimizer.optimizationHistory.n_rows, N );


}

TEST_F(OptimizationTest, doesObjectiveFunctionHaveGradients){

	prepareObjectiveFunctionWithAdjoint();
	ASSERT_TRUE(testOptimizer.doesObjectiveFunctionHaveGradients());

}

TEST_F(OptimizationTest, setGradientGlobalOptimum){

	prepareObjectiveFunctionWithAdjoint();

	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();

	testOptimizer.setOptimizationHistory();

	testOptimizer.setGradientGlobalOptimum();

	ASSERT_TRUE(testOptimizer.globalOptimalDesign.gradient.size() > 0);

}



TEST_F(OptimizationTest, changeSettingsForAGradientBasedStep){

	prepareObjectiveFunctionWithAdjoint();

	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();

	testOptimizer.setOptimizationHistory();

	testOptimizer.setGradientGlobalOptimum();
	testOptimizer.changeSettingsForAGradientBasedStep();

	ASSERT_TRUE(testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep.size() == 2);
	ASSERT_TRUE(testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep.size() == 2);


}

TEST_F(OptimizationTest, checkIfDesignIsWithinBounds){

	rowvec x(3);
	x(0) =  0.2; x(1) = 0.15; x(2) = 0.22;

	testOptimizer.dimension = 3;
	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);

	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.1);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.25);


	bool flag = testOptimizer.checkIfDesignIsWithinBounds(x);
	ASSERT_TRUE(flag);
	x(0) =  0.2; x(1) = 0.15; x(2) = 0.27;
	flag = testOptimizer.checkIfDesignTouchesBounds(x);
	ASSERT_FALSE(flag);

}


TEST_F(OptimizationTest, checkIfDesignTouchesBounds){

	rowvec x(3);
	x(0) =  0.2; x(1) = 0.15; x(2) = 0.25;

	testOptimizer.dimension = 3;
	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);

	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.1);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.25);


	bool flag = testOptimizer.checkIfDesignTouchesBounds(x);
	ASSERT_TRUE(flag);
	x(0) =  0.2; x(1) = 0.15; x(2) = 0.2;
	flag = testOptimizer.checkIfDesignTouchesBounds(x);
	ASSERT_FALSE(flag);

}




TEST_F(OptimizationTest, determineMaxStepSizeForGradientStep){

	rowvec x(3);
	rowvec g(3);

	x(0) =  0.2; x(1) = 0.15; x(2) = 0.20;
	g(0) = -0.2; g(1) = 0.01; g(2) = -0.001;

	testOptimizer.dimension = 3;
	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep = zeros<vec>(3);

	testOptimizer.lowerBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.1);
	testOptimizer.upperBoundsForAcqusitionFunctionMaximizationGradientStep.fill(0.25);
	double maxStep = testOptimizer.determineMaxStepSizeForGradientStep(x,g);

	EXPECT_GT(maxStep,10E-08);


}




TEST_F(OptimizationTest, findTheMostPromisingDesignGradientStep){

	prepareObjectiveFunctionWithAdjoint();

	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();

	testOptimizer.setOptimizationHistory();

	testOptimizer.findTheMostPromisingDesignGradientStep();

	ASSERT_TRUE(testOptimizer.theMostPromisingDesigns.size() == 1);


}

//TEST_F(OptimizationTest, EGOUnconstrainedWithGradientEnhancedModel){
//
//	prepareObjectiveFunctionWithAdjoint();
//
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(20);
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 0.01);
//
//	abort();
//
//}


//TEST_F(OptimizationTest, EGOconstrainedWithGradientEnhancedModel){
//
//	prepareObjectiveFunctionWithAdjoint();
//	prepareFirstConstraint();
//
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(20);
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 0.01);
//
//
//
//}









//TEST_F(OptimizationTest, EGOUnconstrained){
//
//	prepareObjectiveFunction();
//
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(10);
//
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 20.0);
//
//}







//TEST_F(OptimizationTest, EGOUnconstrainedWithTangentEnhancedModel){
//
//	prepareObjectiveFunctionWithTangent();
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(10);
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 20.0);
//
//
//}


//TEST_F(OptimizationTest, EGOUnconstrainedWithMLLowFiAdjoint){
//
//	prepareObjectiveFunctionWithMLLowFiAdjoint();
//
//
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(10);
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 25.0);
//
//
//}



//TEST_F(OptimizationTest, EGOUnconstrainedWithML){
//
//	prepareObjectiveFunctionWithML();
//
//
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	//	testOptimizer.setDisplayOn();
//	testOptimizer.setMaximumNumberOfIterations(10);
//	testOptimizer.performEfficientGlobalOptimization();
//
//	mat results;
//	results.load("himmelblau.csv", csv_ascii);
//
//	vec objectiveFunctionValues = results.col(2);
//
//	double minObjFun = min(objectiveFunctionValues);
//
//	EXPECT_LT(minObjFun, 25.0);
//
//}

TEST_F(OptimizationTest, updateOptimizationHistory){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();

	testOptimizer.setOptimizationHistory();

	Design d(2);
	rowvec dv(2); dv(0) = 4.5; dv(1) = 0.5;
	d.designParameters = dv;
	d.trueValue = -10;
	d.numberOfConstraints = 2;
	rowvec constraintVals(2);
	constraintVals(0) = 100;
	constraintVals(1) = 10;
	d.constraintTrueValues = constraintVals;
	d.improvementValue = 9;
	d.isDesignFeasible = false;

	testOptimizer.updateOptimizationHistory(d);


	unsigned int N = himmelblauFunction.function.numberOfTrainingSamples;
	ASSERT_EQ(testOptimizer.optimizationHistory.n_rows, N+1 );

}






TEST_F(OptimizationTest, calculateFeasibilityProbabilities){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();
	testOptimizer.trainSurrogates();

	DesignForBayesianOptimization design(2,2);
	rowvec dv(2);
	dv(0) = 0.4;
	dv(1) = 0.28;
	design.dv = dv;

	testOptimizer.estimateConstraints(design);

	rowvec constraintValues = design.constraintValues;
	rowvec dvNotNormalized =normalizeVectorBack(design.dv, lb, ub);

	testOptimizer.calculateFeasibilityProbabilities(design);
	ASSERT_LT(design.constraintFeasibilityProbabilities(0), 1.1);
	ASSERT_LT(design.constraintFeasibilityProbabilities(1), 1.1);


}

TEST_F(OptimizationTest, estimateConstraints){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();
	testOptimizer.trainSurrogates();

	DesignForBayesianOptimization design(2,2);
	rowvec dv(2);
	dv(0) = 0.1;
	dv(1) = 0.2;
	design.dv = dv;

	testOptimizer.estimateConstraints(design);

	rowvec constraintValues = design.constraintValues;
	rowvec dvNotNormalized =normalizeVectorBack(design.dv, lb, ub);

	double constraintVal = constraint1.function.func_ptr(dvNotNormalized.memptr());

	double error = fabs(constraintVal -constraintValues(0) );
	EXPECT_LT(error, 1.0);

	constraintVal = constraint2.function.func_ptr(dvNotNormalized.memptr());
	error = fabs(constraintVal -constraintValues(1) );
	EXPECT_LT(error, 1.0);


}






//TEST_F(OptimizationTest, findTheMostPromisingDesignWithConstraint){
//
//	prepareObjectiveFunction();
//	prepareFirstConstraint();
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	testOptimizer.initializeSurrogates();
//	testOptimizer.trainSurrogates();
//
//	testOptimizer.findTheMostPromisingDesignEGO();
//
//	DesignForBayesianOptimization design = testOptimizer.getDesignWithMaxExpectedImprovement();
//
//	double dv1 = design.dv(0);
//	double dv2 = design.dv(1);
//
//	ASSERT_LT(dv1,0.5);
//	ASSERT_GT(dv1,0.0);
//	ASSERT_LT(dv2,0.5);
//	ASSERT_GT(dv2,0.0);
//
//}

TEST_F(OptimizationTest, setOptimizationProblem){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);


	ASSERT_TRUE(testOptimizer.numberOfConstraints == 2);
	ASSERT_TRUE(testOptimizer.ifBoxConstraintsSet);
	ASSERT_TRUE(testOptimizer.ifObjectFunctionIsSpecied);
	ASSERT_TRUE(testOptimizer.isConstrained());

}

TEST_F(OptimizationTest, initializeSurrogates){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();

	ASSERT_TRUE(testOptimizer.ifSurrogatesAreInitialized);

}


//TEST_F(OptimizationTest, trainSurrogates){
//
//	prepareObjectiveFunction();
//	prepareFirstConstraint();
//	prepareSecondConstraint();
//	testOptimizer.setBoxConstraints(boxConstraints);
//
//	testOptimizer.initializeSurrogates();
//	testOptimizer.trainSurrogates();
//
//	ASSERT_TRUE(testOptimizer.ifSurrogatesAreInitialized);
//
//
//}






