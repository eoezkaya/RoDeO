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
#include "auxiliary_functions.hpp"
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

TEST_F(EAPopulationTest, testgetIndividualIdInPopulationByOrder){

	addSomeInvididualsToTestPopulation(3);
	unsigned int id = testPopulation.getIndividualIdInPopulationByOrder(1);

	ASSERT_TRUE(id == 1);

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



TEST_F(EAPopulationTest, testpickUpARandomIndividualForReproduction){

	addSomeInvididualsToTestPopulation(20);

	testPopulation.updatePopulationProperties();

	unsigned int count = 0;
	for(unsigned int i=0; i<1000; i++){

		EAIndividual random = testPopulation.pickUpARandomIndividualForReproduction();
		if(random.getId() == 0) count++;

	}

	EXPECT_TRUE(count>100);

}

TEST_F(EAPopulationTest, testpickUpAnIndividualThatWillDie){

	addSomeInvididualsToTestPopulation(20);

	testPopulation.updatePopulationProperties();

	unsigned int countEvent = 0;
	for(unsigned int i=0; i<1000; i++){

		unsigned int order = testPopulation.pickUpAnIndividualThatWillDie();
		if(order == 19) countEvent++;

	}

	EXPECT_TRUE(countEvent>100);

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

	EAOptimizer testOptimizer;


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

	unsigned int NRandomEvents = 10000;
	unsigned int populationSize = 20;
	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();

	EAIndividual best = testOptimizer.getSolution();


	unsigned int eventCount=0;
	for(unsigned int i=0; i<NRandomEvents; i++){

		std::pair<EAIndividual, EAIndividual> testPair = testOptimizer.generateRandomParents();

		if(testPair.first.getId() == best.getId()) eventCount++;

	}

	EXPECT_TRUE(eventCount>2*NRandomEvents/populationSize);


}

TEST_F(EAOptimizerTest, testcrossOver){

	unsigned int N = 4;
	unsigned int populationSize = 20;
	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();


	std::pair<EAIndividual, EAIndividual> testPair = testOptimizer.generateRandomParents();

	testOptimizer.setMutationProbability(0.0000000001);

	EAIndividual child = testOptimizer.crossOver(testPair);

	vec childGenes  = child.getGenes();
	vec motherGenes = testPair.first.getGenes();
	vec fatherGenes = testPair.second.getGenes();

	for(unsigned int i=0;i<childGenes.size();i++){

		ASSERT_TRUE(isBetween(childGenes(i), motherGenes(i),  fatherGenes(i)));


	}



}

TEST_F(EAOptimizerTest, testcrossOverWithLotsOfMutations){

	unsigned int N = 4;
	unsigned int NRandomEvents = 1000;
	unsigned int populationSize = 20;
	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();

	testOptimizer.setMutationProbability(0.2);

	unsigned int eventCount = 0;
	for(unsigned int i=0; i<NRandomEvents; i++){

		std::pair<EAIndividual, EAIndividual> testPair = testOptimizer.generateRandomParents();
		EAIndividual child = testOptimizer.crossOver(testPair);


		vec childGenes  = child.getGenes();
		vec motherGenes = testPair.first.getGenes();
		vec fatherGenes = testPair.second.getGenes();

		for(unsigned int j=0;j<childGenes.size();j++){

			if(!isBetween(childGenes(j), motherGenes(j),  fatherGenes(j))){


				eventCount++;
			}


		}


	}

	EXPECT_TRUE(eventCount>200);

}



TEST_F(EAOptimizerTest, testapplyMutation){


	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);

	vec inputGenes = testBounds.generateVectorWithinBounds();
	vec inputGenesSave = inputGenes;

	double almostCertainProbabilty = 0.9999999999999;
	testOptimizer.setMutationProbability(almostCertainProbabilty);
	testOptimizer.applyMutation(inputGenes);


	double distance =  fabs(inputGenes(0) - inputGenesSave(0));
	EXPECT_GT(distance, 10E-6);

	double almostZeroProbability = 0.0000000000001;
	testOptimizer.setMutationProbability(almostZeroProbability);
	inputGenes = inputGenesSave;
	testOptimizer.applyMutation(inputGenes);


	distance =  fabs(inputGenes(0) - inputGenesSave(0));
	EXPECT_LT(distance, 10E-6);


}


TEST_F(EAOptimizerTest, testaddNewIndividualsToPopulation){

	unsigned int populationSize = 50;
	unsigned int sizeOfNewGenereration = 100;
	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();

	testOptimizer.setMutationProbability(0.2);

	testOptimizer.setNumberOfNewIndividualsInAGeneration(100);
	testOptimizer.addNewIndividualsToPopulation();

	ASSERT_TRUE(testOptimizer.getPopulationSize() == (populationSize + sizeOfNewGenereration));

}

TEST_F(EAOptimizerTest, testaddNewIndividualsToPopulationParallel){

	unsigned int populationSize = 50;
	unsigned int sizeOfNewGenereration = 1000;
	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();

	testOptimizer.setMutationProbability(0.2);
	testOptimizer.setNumberOfThreads(4);

	testOptimizer.setNumberOfNewIndividualsInAGeneration(sizeOfNewGenereration);
	testOptimizer.addNewIndividualsToPopulation();


	ASSERT_TRUE(testOptimizer.getPopulationSize() == (populationSize + sizeOfNewGenereration));


}
TEST_F(EAOptimizerTest, testremoveIndividualsFromPopulation){

	unsigned int populationSize = 50;
	unsigned int numberOfDeaths = 10;

	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();


	testOptimizer.setNumberOfDeathsInAGeneration(numberOfDeaths);
	testOptimizer.removeIndividualsFromPopulation();

	ASSERT_TRUE(testOptimizer.getPopulationSize() == (populationSize - numberOfDeaths));

}



TEST_F(EAOptimizerTest, testgenerateNewGeneration){

	unsigned int populationSize = 50;
	unsigned int sizeOfNewGenereration = 100;
	unsigned int numberOfDeaths = 10;

	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(sizeOfNewGenereration);
	testOptimizer.setNumberOfDeathsInAGeneration(numberOfDeaths);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.initializePopulation();

	testOptimizer.generateANewGeneration();

	ASSERT_TRUE(testOptimizer.getPopulationSize() == (populationSize + sizeOfNewGenereration - numberOfDeaths));


}
TEST_F(EAOptimizerTest, testoptimize){

	unsigned int populationSize = 100;
	unsigned int sizeOfNewGenereration = 100;
	unsigned int numberOfDeaths = 50;

	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(sizeOfNewGenereration);
	testOptimizer.setNumberOfDeathsInAGeneration(numberOfDeaths);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.setNumberOfGenerations(20);
	testOptimizer.setMaximumNumberOfGeneratedIndividuals(2000);


	testOptimizer.optimize();



}


TEST_F(EAOptimizerTest, testwriteAndReadWarmRestartFile){

	unsigned int populationSize = 10;


	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.initializePopulation();

	testOptimizer.setFilenameWarmStart("EggholderWarmStart.csv");
	testOptimizer.writeWarmRestartFile();
	testOptimizer.resetPopulation();

	unsigned int size = testOptimizer.getPopulationSize();
	ASSERT_EQ(size,0);


	testOptimizer.readWarmRestartFile();
	size = testOptimizer.getPopulationSize();
	ASSERT_EQ(size,populationSize);

	remove("EggholderWarmStart.csv");
}


TEST_F(EAOptimizerTest, testoptimizeWithWarmStart){

	unsigned int populationSize = 100;
	unsigned int sizeOfNewGenereration = 100;
	unsigned int numberOfDeaths = 50;

	testOptimizer.setInitialPopulationSize(populationSize);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(sizeOfNewGenereration);
	testOptimizer.setNumberOfDeathsInAGeneration(numberOfDeaths);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.setNumberOfGenerations(3);
	testOptimizer.setMaximumNumberOfGeneratedIndividuals(2000);
	testOptimizer.setFilenameWarmStart("EggholderWarmStart.csv");
//	testOptimizer.setDisplayOn();

	testOptimizer.optimize();


	testOptimizer.setWarmStartOn();
	testOptimizer.writeWarmRestartFile();
	testOptimizer.resetPopulation();

	testOptimizer.optimize();
	remove("EggholderWarmStart.csv");

}

/* Using this class, we test the derived class functionality based on the base class "EAOptimizer" */
class EggholderOptimizer : public EAOptimizer{

private:



public:

	double calculateObjectiveFunctionInternal(vec& input){

			return Eggholder(input);

		}



};

class EAOptimizerTestWithDerivedClass : public ::testing::Test {
protected:
	void SetUp() override {

		testOptimizer.setDimension(2);
		Bounds testBounds(2);
		testBounds.setBounds(0.0,512.0);

		testOptimizer.setBounds(testBounds);
		testOptimizer.setProblemName("Eggholder_Minimization");




	}

	//  void TearDown() override {}

	EggholderOptimizer testOptimizer;


};

TEST_F(EAOptimizerTestWithDerivedClass, testcalculateObjectiveFunctionInternal){

	vec x(2); x(0) = 10.0; x(1) = 15.0;
	double fx = testOptimizer.calculateObjectiveFunctionInternal(x);

	double error = fabs(Eggholder(x) - fx);
	EXPECT_LT(error,10E-8);

}


TEST_F(EAOptimizerTestWithDerivedClass, testOptimizeWithDerivedClass){



	testOptimizer.setInitialPopulationSize(50);
	testOptimizer.setNumberOfNewIndividualsInAGeneration(100);
	testOptimizer.setNumberOfDeathsInAGeneration(50);
	testOptimizer.setMutationProbability(0.1);
	testOptimizer.setNumberOfGenerations(10);
	testOptimizer.setMaximumNumberOfGeneratedIndividuals(1000);

	testOptimizer.optimize();


	EAIndividual objectiveFunctionValueOptimized = testOptimizer.getSolution();

	ASSERT_LT(objectiveFunctionValueOptimized.getObjectiveFunctionValue(), -500.0);


}






#endif
