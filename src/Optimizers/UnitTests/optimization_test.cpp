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


		constraintFunc1.setDimension(2);
		constraintFunc2.setDimension(2);


		constraintDefinition1.designVectorFilename = "dv.dat";
		constraintDefinition1.executableName = "constraint1";
		constraintDefinition1.outputFilename = "constraintFunction1.dat";

		std::string defitinition1 = "constraint1 < 10.0";
		constraintDefinition1.setDefinition(defitinition1);
		constraintDefinition1.nameHighFidelityTrainingData = "constraint1.csv";


		constraintDefinition2.designVectorFilename = "dv.dat";
		constraintDefinition2.executableName = "constraint2";
		constraintDefinition2.outputFilename = "constraintFunction2.dat";

		std::string defitinition2 = "constraint2 > 3.0";
		constraintDefinition2.setDefinition(defitinition2);
		constraintDefinition2.nameHighFidelityTrainingData = "constraint2.csv";

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

		compileWithCpp("himmelblau.cpp", definition.executableName);

		himmelblauFunction.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}

	void prepareObjectiveFunctionWithAdjoint(void){

		compileWithCpp("himmelblauAdjoint.cpp", definition.executableName);

		himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
		definition.modelHiFi = GRADIENT_ENHANCED;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}




	void prepareObjectiveFunctionWithTangent(void){

		compileWithCpp("himmelblauTangent.cpp", definition.executableName);

		himmelblauFunction.function.generateTrainingSamplesWithTangents();
		definition.modelHiFi = TANGENT;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);


	}

	void prepareObjectiveFunctionWithML(void){

		compileWithCpp("himmelblau.cpp", definition.executableName);
		compileWithCpp("himmelblauLowFidelity.cpp", definition.executableNameLowFi);
		himmelblauFunction.function.generateTrainingSamplesMultiFidelity();

		definition.ifMultiLevel = true;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);
	}

	void prepareObjectiveFunctionWithMLLowFiAdjoint(void){

		compileWithCpp("himmelblau.cpp", definition.executableName);
		compileWithCpp("himmelblauAdjointLowFi.cpp", definition.executableNameLowFi);
		himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();

		definition.ifMultiLevel = true;
		definition.modelHiFi = ORDINARY_KRIGING;
		definition.modelLowFi = GRADIENT_ENHANCED;
		objFunHimmelblau.setParametersByDefinition(definition);
		testOptimizer.addObjectFunction(objFunHimmelblau);
	}



	void prepareFirstConstraint(void){

		compileWithCpp("constraint1.cpp", constraintDefinition1.executableName);

		mat X = himmelblauFunction.function.trainingSamplesInput;
		X = shuffleRows(X);
		mat Xreduced = X.submat(0, 0, constraint1.function.numberOfTrainingSamples -1, 1);
		constraint1.function.trainingSamplesInput = Xreduced;
		constraint1.function.ifInputSamplesAreGenerated = true;
		constraint1.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);
		constraintFunc1.setParametersByDefinition(constraintDefinition1);
		constraintFunc1.setID(0);
		testOptimizer.addConstraint(constraintFunc1);

	}
	void prepareSecondConstraint(void){

		compileWithCpp("constraint2.cpp", constraintDefinition2.executableName);
		mat X = himmelblauFunction.function.trainingSamplesInput;
		X = shuffleRows(X);
		mat Xreduced = X.submat(0, 0, constraint2.function.numberOfTrainingSamples -1, 1);
		constraint2.function.trainingSamplesInput = Xreduced;
		constraint2.function.ifInputSamplesAreGenerated = true;
		constraint2.function.generateTrainingSamples();
		objFunHimmelblau.setParametersByDefinition(definition);
		constraintFunc2.setParametersByDefinition(constraintDefinition2);
		constraintFunc2.setID(1);
		testOptimizer.addConstraint(constraintFunc2);

	}

	void TearDown() override {



	}

	std::string problemName = "himmelblauOptimization";
	Optimizer testOptimizer;
	ObjectiveFunction objFunHimmelblau;
	ConstraintFunction constraintFunc1;
	ConstraintFunction constraintFunc2;

	HimmelblauFunction himmelblauFunction;
	HimmelblauConstraintFunction1 constraint1;
	HimmelblauConstraintFunction2 constraint2;

	ObjectiveFunctionDefinition definition;
	ConstraintDefinition constraintDefinition1;
	ConstraintDefinition constraintDefinition2;

	vec lb;
	vec ub;

	Bounds boxConstraints;


};


TEST_F(OptimizationTest, constructor){

	ASSERT_FALSE(testOptimizer.ifBoxConstraintsSet);
	ASSERT_FALSE(testOptimizer.ifObjectFunctionIsSpecied);
	ASSERT_FALSE(testOptimizer.ifConstrained());
}

TEST_F(OptimizationTest, setBoxConstraints){

	prepareObjectiveFunction();
	testOptimizer.setBoxConstraints(boxConstraints);

	ASSERT_TRUE(testOptimizer.ifBoxConstraintsSet);


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
	mat optimizationHistory = testOptimizer.getOptimizationHistory();

	unsigned int N = himmelblauFunction.function.numberOfTrainingSamples;
	ASSERT_EQ(optimizationHistory.n_rows, N );


}

TEST_F(OptimizationTest, EGOUnconstrainedWithMLLowFiAdjoint){

	prepareObjectiveFunctionWithMLLowFiAdjoint();


	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.setDisplayOn();
	testOptimizer.setMaximumNumberOfIterations(50);
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 5.0);

	abort();


}



TEST_F(OptimizationTest, EGOUnconstrainedWithML){

	prepareObjectiveFunctionWithML();


	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.setDisplayOn();
	testOptimizer.setMaximumNumberOfIterations(50);
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 5.0);

}









TEST_F(OptimizationTest, EGOUnconstrainedWithGGEKModel){

	prepareObjectiveFunctionWithAdjoint();

	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.setDisplayOn();
	testOptimizer.setMaximumNumberOfIterations(50);
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 5.0);

}








TEST_F(OptimizationTest, EGOUnconstrained){

	abort();

	prepareObjectiveFunction();

	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.setDisplayOn();
	testOptimizer.setZoomInOn();
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 5.0);

	abort();


}



TEST_F(OptimizationTest, EGOUnconstrainedWithTangent){

	prepareObjectiveFunctionWithTangent();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.setDisplayOn();
	testOptimizer.setMaximumNumberOfIterations(50);
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 5.0);

	abort();


}






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

	mat optimizationHistory = testOptimizer.getOptimizationHistory();

	unsigned int N = himmelblauFunction.function.numberOfTrainingSamples;
	ASSERT_EQ(optimizationHistory.n_rows, N+1 );

}


TEST_F(OptimizationTest, reduceTrainingDataFiles){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();
	testOptimizer.setOptimizationHistory();
	testOptimizer.setZoomInOn();
	testOptimizer.setZoomFactor(0.5);
	//	testOptimizer.setDisplayOn();
	testOptimizer.zoomInDesignSpace();
	testOptimizer.reduceBoxConstraints();


	mat readData;
	readData.load(definition.nameHighFidelityTrainingData, csv_ascii);

	unsigned int N1 = readData.n_rows;


	testOptimizer.reduceTrainingDataFiles();
	readData.load(definition.nameHighFidelityTrainingData, csv_ascii);

	unsigned int N2 = readData.n_rows;
	EXPECT_TRUE(N2 < N1);

}




TEST_F(OptimizationTest, EGOConstrained){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();

	testOptimizer.setZoomInOn();
	testOptimizer.setMaximumNumberOfIterations(100);
	testOptimizer.setHowOftenZoomIn(20);
	//	testOptimizer.setDisplayOn();
	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.EfficientGlobalOptimization();

	mat results;
	results.load("himmelblau.csv", csv_ascii);

	vec objectiveFunctionValues = results.col(2);

	double minObjFun = min(objectiveFunctionValues);

	EXPECT_LT(minObjFun, 10.0);

}








TEST_F(OptimizationTest, zoomInDesignSpace){

	prepareObjectiveFunction();
	prepareFirstConstraint();

	testOptimizer.setBoxConstraints(boxConstraints);
	testOptimizer.initializeSurrogates();
	testOptimizer.clearOptimizationHistoryFile();
	testOptimizer.prepareOptimizationHistoryFile();
	testOptimizer.setOptimizationHistory();

	testOptimizer.zoomInDesignSpace();
	testOptimizer.zoomInDesignSpace();

	pair<vec,vec> bounds = testOptimizer.getBoundsForAcqusitionFunctionMaximization();

	vec lb = bounds.first;
	vec ub = bounds.second;



	Design optimalDesign = testOptimizer.getGlobalOptimalDesign();

	double J = optimalDesign.trueValue;

	EXPECT_TRUE( ub(0) - lb(0) < 0.5 );
	EXPECT_TRUE( ub(1) - lb(1) < 0.5 );
	EXPECT_TRUE( J < 500.0 );
	EXPECT_TRUE( J > 0.0 );

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


	double constraintVal1 = constraint1.function.func_ptr(dvNotNormalized.memptr());
	double constraintVal2 = constraint2.function.func_ptr(dvNotNormalized.memptr());

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


TEST_F(OptimizationTest, findTheMostPromisingDesignWithConstraint){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();
	testOptimizer.trainSurrogates();

	testOptimizer.findTheMostPromisingDesign();

	DesignForBayesianOptimization design = testOptimizer.getDesignWithMaxExpectedImprovement();

	double dv1 = design.dv(0);
	double dv2 = design.dv(1);

	ASSERT_LT(dv1,0.5);
	ASSERT_GT(dv1,0.0);
	ASSERT_LT(dv2,0.5);
	ASSERT_GT(dv2,0.0);



}





TEST_F(OptimizationTest, setOptimizationProblem){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);


	ASSERT_TRUE(testOptimizer.numberOfConstraints == 2);
	ASSERT_TRUE(testOptimizer.ifBoxConstraintsSet);
	ASSERT_TRUE(testOptimizer.ifObjectFunctionIsSpecied);
	ASSERT_TRUE(testOptimizer.ifConstrained());



}

TEST_F(OptimizationTest, initializeSurrogates){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();

	ASSERT_TRUE(testOptimizer.ifSurrogatesAreInitialized);

}






TEST_F(OptimizationTest, trainSurrogates){

	prepareObjectiveFunction();
	prepareFirstConstraint();
	prepareSecondConstraint();
	testOptimizer.setBoxConstraints(boxConstraints);

	testOptimizer.initializeSurrogates();
	testOptimizer.trainSurrogates();

	ASSERT_TRUE(testOptimizer.ifSurrogatesAreInitialized);


}






//
//TEST(testOptimizer, testTangentEnhancedOptimization){
//
//	chdir("./testTangentEnhancedOptimization");
//	compileWithCpp("himmelblauWithTangent.cpp", "himmelblauWithTangent");
//
//	Bounds boxConstraints(2);
//	boxConstraints.setBounds(-6.0,6.0);
//
//	std::string studyName = "testOptimizerTangentEnhanced";
//	Optimizer testStudy(studyName, 2);
//	testStudy.setBoxConstraints(boxConstraints);
//	testStudy.setMaximumNumberOfIterations(100);
//
//
//
//
//
//
//
//	abort();
//}
//
//
//
//TEST(testOptimizer, testMLOptimization){
//
//	chdir("./testMultiLevelOptimization");
//	compileWithCpp("himmelblauHighFidelity.cpp", "himmelblauHighFidelity");
//	compileWithCpp("himmelblauLowFidelity.cpp",  "himmelblauLowFidelity");
//
//
//	Bounds boxConstraints(2);
//    boxConstraints.setBounds(-6.0,6.0);
//
//	unsigned int NhiFi = 100;
//
//	generateHimmelblauDataMultiFidelity("HimmelblauHiFiData.csv", "HimmelblauLowFiData.csv", 50, 100);
//
//
//	std::string studyName = "testOptimizerMultiLevel";
//	Optimizer testStudy(studyName, 2);
//
//	testStudy.setMaximumNumberOfIterations(100);
//	testStudy.setMaximumNumberOfIterationsLowFidelity(100);
//
//
//
//	testStudy.setBoxConstraints(boxConstraints);
//
//
//	ObjectiveFunction objFunTest("testObjectiveFunctionMLSurrogate",2);
//
//
//	objFunTest.setParameterBounds(boxConstraints);
//
//	ObjectiveFunctionDefinition testObjectiveFunctionDef("testObjectiveFunctionMLSurrogate");
//	testObjectiveFunctionDef.outputFilename      = "objFunVal.dat";
//	testObjectiveFunctionDef.outputFilenameLowFi = "objFunVal.dat";
//	testObjectiveFunctionDef.ifMultiLevel = true;
//	testObjectiveFunctionDef.designVectorFilename = "dv.dat";
//	testObjectiveFunctionDef.executableName = "himmelblauHighFidelity";
//	testObjectiveFunctionDef.executableNameLowFi = "himmelblauLowFidelity";
//	testObjectiveFunctionDef.nameHighFidelityTrainingData = "HimmelblauHiFiData.csv";
//	testObjectiveFunctionDef.nameLowFidelityTrainingData  = "HimmelblauLowFiData.csv";
//
//
//	objFunTest.setParametersByDefinition(testObjectiveFunctionDef);
//
//	testStudy.addObjectFunction(objFunTest);
//	testStudy.setFileNameDesignVector("dv.dat");
//
//	testStudy.setDisplayOn();
//
//	testStudy.setHowOftenTrainModels(1000);
//
//	testStudy.EfficientGlobalOptimization2();
//
//	chdir("../");
//
//	abort();
//}
//
//
//
//
//
//
//
//
//
//

//
//
//TEST(testOptimizer, testMaximizeEIGradientBased){
//
//	mat samples(10,3);
//
//
//	 samples(0,0) = 1.4199;  samples(0,1) = 0.2867; samples(0,2) = 2.0982;
//	 samples(1,0) = 1.2761;  samples(1,1) = 0.7363; samples(1,2) = 2.1706;
//	 samples(2,0) =-4.8526;  samples(2,1) =-1.3832; samples(2,2) = 25.4615;
//	 samples(3,0) =-4.9643;  samples(3,1) = 2.5681; samples(3,2) = 31.2395;
//	 samples(4,0) =-3.2966;  samples(4,1) = 3.7140; samples(4,2) = 24.6612;
//	 samples(5,0) = 2.1202;  samples(5,1) = 1.5686; samples(5,2) =  6.9559;
//	 samples(6,0) = 4.0689;  samples(6,1) = 3.8698; samples(6,2) = 31.5311;
//	 samples(7,0) = 3.6139;  samples(7,1) =-0.5469; samples(7,2) = 13.3596;
//	 samples(8,0) = 4.5835;  samples(8,1) =-3.9091; samples(8,2) = 36.2896;
//	 samples(9,0) = 3.8641;  samples(9,1) = 2.3025; samples(9,2) = 20.2327;
//
//	vec lb(2); lb.fill(-5.0);
//	vec ub(2); ub.fill(5.0);
//
//
//	saveMatToCVSFile(samples,"ObjFuncTest.csv");
//
//	ObjectiveFunction objFunc("ObjFuncTest", 2);
//	objFunc.setParameterBounds(lb,ub);
//
//	ObjectiveFunctionDefinition testObjectiveFunctionDef("ObjectiveFunctionTest");
//	objFunc.setParametersByDefinition(testObjectiveFunctionDef);
//
//	objFunc.initializeSurrogate();
//
//
//	std::string studyName = "testOptimizer";
//	Optimizer testStudy(studyName, 2);
//	testStudy.addObjectFunction(objFunc);
//	testStudy.initializeSurrogates();
//
//	rowvec dv(2); dv(0) = 0.29; dv(1) = 0.29;
//
//	CDesignExpectedImprovement initialDesign(dv);
//
//
//	CDesignExpectedImprovement optimizedDesign = testStudy.MaximizeEIGradientBased(initialDesign);
//
//	ASSERT_GT(optimizedDesign.valueExpectedImprovement, initialDesign.valueExpectedImprovement + 1.0);
//
//
//}



