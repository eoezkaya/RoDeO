/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022. Chair for Scientific Computing (SciComp), TU Kaiserslautern
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


#include "ea_optimizer.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>

#define TESTEAOPTIMIZER
#ifdef  TESTEAOPTIMIZER



TEST(testEAOptimizer, testEAIndividual_Constructor){

	EAIndividual testIndividual(2);

	ASSERT_TRUE(testIndividual.getDimension() == 2);
	ASSERT_TRUE(testIndividual.getId() == 0);


}




TEST(testEAOptimizer, testEAIndividual_initializeGenes){

	EAIndividual testIndividual(7);


	vec genesInput(7,fill::randu);

	testIndividual.initializeGenes(genesInput);

	vec genesOutput = testIndividual.getGenes();

	ASSERT_TRUE(genesOutput(3) == genesInput(3));

}







class EAPopulationTest : public ::testing::Test {
protected:
	void SetUp() override {


		testPopulation.setDimension(2);



	}

	//  void TearDown() override {}

	EAPopulation testPopulation;


	void addSomeInvididualsToTestPopulation(unsigned int howMany){

		for(unsigned int i=0; i<howMany; i++){

			vec dv(2);
			EAIndividual testIndividual(2);
			dv(0) = i+1; dv(1) = i-1;
			testIndividual.setId(i);
			testIndividual.setGenes(dv);
			testIndividual.setObjectiveFunctionValue(2.0*i);
			testPopulation.addIndividual(testIndividual);

		}

	}


};




TEST_F(EAPopulationTest, testConstructor){


	ASSERT_TRUE(testPopulation.getSize() == 0);

}



TEST_F(EAPopulationTest, testaddInvididual){

	EAIndividual testIndividual(2);
	testPopulation.addIndividual(testIndividual);

	ASSERT_TRUE(testPopulation.getSize() == 1);

}


TEST_F(EAPopulationTest, testaddAGroupOfIndividuals){

	std::vector<EAIndividual> itemsToAdd;
	addSomeInvididualsToTestPopulation(4);
	EAIndividual testIndividual(2);

	testIndividual.initializeRandom();
	itemsToAdd.push_back(testIndividual);
	testIndividual.initializeRandom();
	itemsToAdd.push_back(testIndividual);


	testPopulation.addAGroupOfIndividuals(itemsToAdd);

	ASSERT_TRUE(testPopulation.getSize() == 6);



}




TEST_F(EAPopulationTest, testgetIdOftheIndividual){

	EAIndividual testIndividual(2);
	testIndividual.setId(12);
	testPopulation.addIndividual(testIndividual);

	unsigned int id = testPopulation.getIdOftheIndividual(0);
	ASSERT_TRUE(id == 12);



}

TEST_F(EAPopulationTest, testgetTheBestIndividual){

	addSomeInvididualsToTestPopulation(3);
	ASSERT_TRUE(testPopulation.getSize() == 3);

	EAIndividual best = testPopulation.getTheBestIndividual();

	ASSERT_TRUE(best.getId() == 0);


}

TEST_F(EAPopulationTest, testgetIndividualOrderInPopulationById){

	addSomeInvididualsToTestPopulation(3);
	unsigned int order = testPopulation.getIndividualOrderInPopulationById(1);

	ASSERT_TRUE(order == 1);

}

TEST_F(EAPopulationTest, testremoveIndividual){

	addSomeInvididualsToTestPopulation(3);
	unsigned int idToRemove = 1;

	testPopulation.removeIndividual(idToRemove);
	ASSERT_TRUE(testPopulation.getIndividualOrderInPopulationById(0) == 0);
	ASSERT_TRUE(testPopulation.getIndividualOrderInPopulationById(2) == 1);
	ASSERT_TRUE(testPopulation.getSize() == 2);


}

TEST_F(EAPopulationTest, testremoveIndividualPopulationMaxandMinGetsUpdated){

	addSomeInvididualsToTestPopulation(5);
	unsigned int idToRemove = 4;
	testPopulation.removeIndividual(idToRemove);
	idToRemove = 0;
	testPopulation.removeIndividual(idToRemove);


	EAIndividual min = testPopulation.getTheBestIndividual();

	ASSERT_TRUE(min.getId() == 1);

	EAIndividual max = testPopulation.getTheWorstIndividual();

	ASSERT_TRUE(max.getId() == 3);

}

TEST_F(EAPopulationTest, testupdatePopulationMinAndMax){


	addSomeInvididualsToTestPopulation(10);


	testPopulation.updatePopulationMinAndMax();

	EAIndividual min = testPopulation.getTheBestIndividual();

	ASSERT_TRUE(min.getId() == 0);

	EAIndividual max = testPopulation.getTheWorstIndividual();

	ASSERT_TRUE(max.getId() == 9);


}


TEST_F(EAPopulationTest, testupdatePopulationFitness){


	addSomeInvididualsToTestPopulation(10);
	testPopulation.updateFitnessValues();

	EAIndividual bestFit = testPopulation.getIndividual(0);
	double fitness = bestFit.getFitnessValue();
	double error = fabs(fitness - 1.0);

	ASSERT_TRUE(error< 10E-08);


}


TEST_F(EAPopulationTest, testupdateReproductionProbabilities){

	addSomeInvididualsToTestPopulation(10);
	testPopulation.updateFitnessValues();
	testPopulation.updateReproductionProbabilities();
	EAIndividual bestFit = testPopulation.getIndividual(0);
	double reproductionP = bestFit.getReproductionProbability();

	ASSERT_TRUE(reproductionP > 0.1);

}


TEST_F(EAPopulationTest, testupdateDeathProbabilities){

	addSomeInvididualsToTestPopulation(20);
	testPopulation.updateFitnessValues();
	testPopulation.updateDeathProbabilities();
	EAIndividual bestFit = testPopulation.getIndividual(0);
	double deathP = bestFit.getReproductionProbability();

	ASSERT_TRUE(deathP < 0.0001);

}




class EAOptimizerTest : public ::testing::Test {
protected:
	void SetUp() override {

		testOptimizer.setDimension(2);
		testOptimizer.setObjectiveFunction(Eggholder);
		Bounds testBounds(2);
		testBounds.setBounds(0.0,512.0);

		testOptimizer.setBounds(testBounds);
		testOptimizer.setProblemName("Eggholder_Minimization");




	}

	//  void TearDown() override {}

	EAOptimizer2 testOptimizer;


};


TEST_F(EAOptimizerTest, testIfConstructorWorks) {

	ASSERT_TRUE(testOptimizer.getDimension() == 2);
	ASSERT_TRUE(testOptimizer.areBoundsSet());
	ASSERT_TRUE(testOptimizer.getNumberOfThreads() == 1);
	ASSERT_TRUE(testOptimizer.isObjectiveFunctionSet());
	ASSERT_TRUE(testOptimizer.isProblemNameSet());
	ASSERT_FALSE(testOptimizer.isFilenameOptimizationResultSet());
	ASSERT_FALSE(testOptimizer.isFilenameWarmStartSet());
	ASSERT_FALSE(testOptimizer.isFilenameOptimizationHistorySet());


}


TEST_F(EAOptimizerTest, testcallObjectiveFunction) {

	EAIndividual testIndividual(2);

	vec dv(2);
	dv(0) = 1.0;
	dv(1) = 1.0;
	testIndividual.initializeGenes(dv);


	testOptimizer.callObjectiveFunction(testIndividual);

	double value = testIndividual.getObjectiveFunctionValue();
	double valueToCheck = Eggholder(dv);
	double error = fabs(value - valueToCheck);

	ASSERT_LT(error,10E-10);

}

TEST_F(EAOptimizerTest, testgenerateRandomIndividual){


	EAIndividual testIndvidual = testOptimizer.generateRandomIndividual();

	ASSERT_TRUE(testIndvidual.getDimension() == 2);

}




TEST_F(EAOptimizerTest, testinitializePopulation){


	testOptimizer.setInitialPopulationSize(100);
	testOptimizer.initializePopulation();

	ASSERT_TRUE(testOptimizer.getPopulationSize() == 100);


}

TEST_F(EAOptimizerTest, testinitializePopulationParallel){


	testOptimizer.setInitialPopulationSize(100);
	testOptimizer.setNumberOfThreads(4);
	testOptimizer.initializePopulation();

	ASSERT_TRUE(testOptimizer.getPopulationSize() == 100);


}




TEST_F(EAOptimizerTest, testgenerateRandomParents){


	testOptimizer.setInitialPopulationSize(10);
	testOptimizer.initializePopulation();

	std::pair<EAIndividual, EAIndividual> testPair = testOptimizer.generateRandomParents();


}
//
//
//TEST(testEAOptimizer, testEAOptimizer_getIndividualLocation){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EAOptimizer test;
//	test.setBounds(testBounds);
//
//
//	test.setObjectiveFunction(Eggholder);
//	test.setDimension(2);
//	test.setInitialPopulationSize(10);
//	test.initializePopulation();
//
//	unsigned int testIndividualIndex = test.getIndividualLocation(6);
//	ASSERT_TRUE(testIndividualIndex == 6);
//
//
//}
//
//
//TEST(testEAOptimizer, testEAOptimizer_applyMutation){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EAOptimizer test;
//	test.setBounds(testBounds);
//
//	vec inputGenes = testBounds.generateVectorWithinBounds();
//	vec inputGenesSave = inputGenes;
//
//	double almostCertainProbabilty = 0.9999999999999;
//	test.setMutationProbability(almostCertainProbabilty);
//	test.applyMutation(inputGenes);
//
//
//	double distance =  fabs(inputGenes(0) - inputGenesSave(0));
//	EXPECT_GT(distance, 10E-6);
//
//	double almostZeroProbability = 0.0000000000001;
//	test.setMutationProbability(almostZeroProbability);
//	inputGenes = inputGenesSave;
//	test.applyMutation(inputGenes);
//
//
//	distance =  fabs(inputGenes(0) - inputGenesSave(0));
//	EXPECT_LT(distance, 10E-6);
//
//
//
//}
//
//
//
//TEST(testEAOptimizer, testEAOptimizer_generateIndividualByReproduction){
//
//	unsigned int N = 4;
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EAOptimizer test;
//	test.setBounds(testBounds);
//
//
//	test.setObjectiveFunction(Eggholder);
//	test.setDimension(2);
//	test.setInitialPopulationSize(N);
//	test.initializePopulation();
//	test.updatePopulationProperties();
//
//	std::pair<unsigned int, unsigned int> testPair = test.generateRandomParents();
//
//	test.setMutationProbability(0.5);
//
//	EAIndividual testIndividual = test.generateIndividualByReproduction(testPair );
//
//	unsigned int id = testIndividual.getId();
//
//	EXPECT_EQ(id,N);
//
//
//}
//
//
//TEST(testEAOptimizer, testEAOptimizer_generateNewGeneration){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EAOptimizer test;
//	test.setBounds(testBounds);
//
//
//	test.setObjectiveFunction(Eggholder);
//	test.setDimension(2);
//	test.setInitialPopulationSize(100);
//	test.setNumberOfNewIndividualsInAGeneration(20);
//	test.setNumberOfDeathsInAGeneration(2);
//	test.setMutationProbability(0.1);
//	test.initializePopulation();
//
//	test.updatePopulationProperties();
//
//	test.generateNewGeneration();
//
//
//
//	unsigned int sizeOfPopulation = test.getPopulationSize();
//	EXPECT_EQ(sizeOfPopulation, 118);
//
//}
//
//TEST(testEAOptimizer, testEAOptimizer_optimize){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EAOptimizer test;
//	test.setBounds(testBounds);
//
//
//	test.setObjectiveFunction(Eggholder);
//	test.setDimension(2);
//	test.setInitialPopulationSize(50);
//	test.setNumberOfNewIndividualsInAGeneration(10);
//	test.setNumberOfDeathsInAGeneration(10);
//	test.setMutationProbability(0.1);
//	test.setNumberOfGenerations(50);
//	test.setDisplayOff();
//	test.setMaximumNumberOfGeneratedIndividuals(1000);
//
//	test.optimize();
//
//	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();
//
//	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);
//
//}
//
///* Using this class, we test the derived class functionality based on the base class "EAOptimizer" */
//class EggholderOptimizer : public EAOptimizer{
//
//private:
//
//	double calculateObjectiveFunctionInternal(vec input){
//
//		return Eggholder(input);
//
//	}
//
//
//public:
//
//
//
//
//};
//
//TEST(testEAOptimizer, testEAOptimizer_optimizeWithDerivedClass){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EggholderOptimizer test;
//
//
//	test.setBounds(testBounds);
//
//	test.setDimension(2);
//	test.setInitialPopulationSize(50);
//	test.setNumberOfNewIndividualsInAGeneration(10);
//	test.setNumberOfDeathsInAGeneration(10);
//	test.setMutationProbability(0.1);
//	test.setNumberOfGenerations(50);
//	test.setDisplayOff();
//	test.setMaximumNumberOfGeneratedIndividuals(1000);
//
//	test.optimize();
//
//
//	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();
//
//	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);
//
//
//}
//
//TEST(testEAOptimizer, testEAOptimizer_optimizeWithDerivedClassParallel){
//
//	Bounds testBounds(2);
//	testBounds.setBounds(0.0,512.0);
//	EggholderOptimizer test;
//
//
//	test.setBounds(testBounds);
//
//	test.setDimension(2);
//	test.setInitialPopulationSize(500);
//	test.setNumberOfNewIndividualsInAGeneration(500);
//	test.setNumberOfDeathsInAGeneration(100);
//	test.setMutationProbability(0.1);
//	test.setNumberOfGenerations(50);
//	test.setDisplayOff();
//	test.setMaximumNumberOfGeneratedIndividuals(1000000);
//	test.setNumberOfThreads(4);
//
//	test.optimize();
//
//
//	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();
//
//	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);
//
//
//}




#endif
