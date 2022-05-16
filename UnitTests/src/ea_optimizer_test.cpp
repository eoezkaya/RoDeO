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



TEST(testEAOptimizer, testEAOptimizer_setDimension){

	EAOptimizer test;

	unsigned int dim = 5;
	test.setDimension(dim);

	unsigned int dimCheck = test.getDimension();
	ASSERT_EQ(dim,dimCheck);
}


TEST(testEAOptimizer, testEAOptimizer_Constructor){

	EAOptimizer test;
	ASSERT_TRUE(test.isOptimizationTypeMinimization());
	ASSERT_FALSE(test.areBoundsSet());

}


TEST(testEAOptimizer, testEAOptimizer_setOptimizationType){

	EAOptimizer test;
	test.setOptimizationType("maximization");
	ASSERT_FALSE(test.isOptimizationTypeMinimization());
	ASSERT_TRUE(test.isOptimizationTypeMaximization());

}

TEST(testEAOptimizer, testEAOptimizer_setBounds){

	Bounds testBounds(5);
	testBounds.setBounds(0.0,1.0);
	EAOptimizer test;
	test.setBounds(testBounds);

	ASSERT_TRUE(test.areBoundsSet());


}



TEST(testEAOptimizer, testEAOptimizer_callObjectiveFunction){

	EAIndividual testIndividual(2);
	EAOptimizer test;

	vec dv(2);
	dv(0) = 1.0;
	dv(1) = 1.0;
	testIndividual.initializeGenes(dv);

	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);

	test.callObjectiveFunction(testIndividual);

	double value = testIndividual.getObjectiveFunctionValue();
	double valueToCheck = Eggholder(dv);
	double error = fabs(value - valueToCheck);

	ASSERT_LT(error,10E-10);

}


TEST(testEAOptimizer, testEAOptimizer_generateRandomIndividual){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);
	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);


	EAIndividual testIndvidual = test.generateRandomIndividual();

	ASSERT_TRUE(testIndvidual.getDimension() == 2);

}

TEST(testEAOptimizer, testEAOptimizer_initializePopulation){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(100);
	test.initializePopulation();

	ASSERT_TRUE(test.getPopulationSize() == 100);


}


TEST(testEAOptimizer, testEAOptimizer_initializePopulationInParallel){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(400);
	test.setNumberOfThreads(4);
	test.initializePopulation();


	ASSERT_TRUE(test.getPopulationSize() == 400);


}




TEST(testEAOptimizer, testEAOptimizer_updatePopulationFitness){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(100);
	test.initializePopulation();

	test.updatePopulationFitnessValues();


	vec fitnessValues = test.getPopulationFitnessValues();
	ASSERT_TRUE(fitnessValues.size() == 100);


	vec objectiveFunctionValues = test.getPopulationObjectiveFunctionValues();
	ASSERT_TRUE(objectiveFunctionValues .size() == 100);

	uword minIndexObjectiveFunction = index_min(objectiveFunctionValues);
	uword maxIndexFitness = index_max(fitnessValues);

	ASSERT_EQ(minIndexObjectiveFunction, maxIndexFitness );


}

TEST(testEAOptimizer, testEAOptimizer_updatePopulationReproductionProbabilities){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(100);
	test.initializePopulation();

	test.updatePopulationFitnessValues();
	test.updatePopulationReproductionProbabilities();



	vec reproductionProbabilities = test.getPopulationReproductionProbabilities();
	ASSERT_TRUE(reproductionProbabilities.size() == 100);


	vec objectiveFunctionValues = test.getPopulationObjectiveFunctionValues();
	ASSERT_TRUE(objectiveFunctionValues .size() == 100);

	uword minIndexObjectiveFunction = index_min(objectiveFunctionValues);
	uword maxIndexReproduction      = index_max(reproductionProbabilities );

	ASSERT_EQ(minIndexObjectiveFunction, maxIndexReproduction );

}

TEST(testEAOptimizer, testEAOptimizer_updatePopulationDeathProbabilities){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(100);
	test.initializePopulation();

	test.updatePopulationFitnessValues();
	test.updatePopulationDeathProbabilities();


	vec deathProbabilities = test.getPopulationDeathProbabilities();
	ASSERT_TRUE(deathProbabilities.size() == 100);


	vec objectiveFunctionValues = test.getPopulationObjectiveFunctionValues();
	ASSERT_TRUE(objectiveFunctionValues .size() == 100);

	uword maxIndexObjectiveFunction = index_max(objectiveFunctionValues);
	uword maxIndexDeath      = index_max( deathProbabilities );

	ASSERT_EQ(maxIndexObjectiveFunction, maxIndexDeath );

}


TEST(testEAOptimizer, testEAOptimizer_generateRandomParents){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(10);
	test.initializePopulation();

	test.updatePopulationProperties();

	std::pair<unsigned int, unsigned int> testPair = test.generateRandomParents();

	ASSERT_FALSE(testPair.first == testPair.second);
	ASSERT_TRUE(testPair.first < 10);

}


TEST(testEAOptimizer, testEAOptimizer_getIndividualLocation){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(10);
	test.initializePopulation();

	unsigned int testIndividualIndex = test.getIndividualLocation(6);
	ASSERT_TRUE(testIndividualIndex == 6);


}


TEST(testEAOptimizer, testEAOptimizer_applyMutation){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);

	vec inputGenes = testBounds.generateVectorWithinBounds();
	vec inputGenesSave = inputGenes;

	double almostCertainProbabilty = 0.9999999999999;
	test.setMutationProbability(almostCertainProbabilty);
	test.applyMutation(inputGenes);


	double distance =  fabs(inputGenes(0) - inputGenesSave(0));
	EXPECT_GT(distance, 10E-6);

	double almostZeroProbability = 0.0000000000001;
	test.setMutationProbability(almostZeroProbability);
	inputGenes = inputGenesSave;
	test.applyMutation(inputGenes);


	distance =  fabs(inputGenes(0) - inputGenesSave(0));
	EXPECT_LT(distance, 10E-6);



}



TEST(testEAOptimizer, testEAOptimizer_generateIndividualByReproduction){

	unsigned int N = 4;
	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(N);
	test.initializePopulation();
	test.updatePopulationProperties();

	std::pair<unsigned int, unsigned int> testPair = test.generateRandomParents();

	test.setMutationProbability(0.5);

	EAIndividual testIndividual = test.generateIndividualByReproduction(testPair );

	unsigned int id = testIndividual.getId();

	EXPECT_EQ(id,N);


}


TEST(testEAOptimizer, testEAOptimizer_generateNewGeneration){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(100);
	test.setNumberOfNewIndividualsInAGeneration(20);
	test.setNumberOfDeathsInAGeneration(2);
	test.setMutationProbability(0.1);
	test.initializePopulation();

	test.updatePopulationProperties();

	test.generateNewGeneration();



	unsigned int sizeOfPopulation = test.getPopulationSize();
	EXPECT_EQ(sizeOfPopulation, 118);

}

TEST(testEAOptimizer, testEAOptimizer_optimize){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EAOptimizer test;
	test.setBounds(testBounds);


	test.setObjectiveFunction(Eggholder);
	test.setDimension(2);
	test.setInitialPopulationSize(50);
	test.setNumberOfNewIndividualsInAGeneration(10);
	test.setNumberOfDeathsInAGeneration(10);
	test.setMutationProbability(0.1);
	test.setNumberOfGenerations(50);
	test.setDisplayOff();
	test.setMaximumNumberOfGeneratedIndividuals(1000);

	test.optimize();

	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();

	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);

}

/* Using this class, we test the derived class functionality based on the base class "EAOptimizer" */
class EggholderOptimizer : public EAOptimizer{

private:

	double calculateObjectiveFunctionInternal(vec input){

		return Eggholder(input);

	}


public:




};

TEST(testEAOptimizer, testEAOptimizer_optimizeWithDerivedClass){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EggholderOptimizer test;


	test.setBounds(testBounds);

	test.setDimension(2);
	test.setInitialPopulationSize(50);
	test.setNumberOfNewIndividualsInAGeneration(10);
	test.setNumberOfDeathsInAGeneration(10);
	test.setMutationProbability(0.1);
	test.setNumberOfGenerations(50);
	test.setDisplayOff();
	test.setMaximumNumberOfGeneratedIndividuals(1000);

	test.optimize();


	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();

	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);


}

TEST(testEAOptimizer, testEAOptimizer_optimizeWithDerivedClassParallel){

	Bounds testBounds(2);
	testBounds.setBounds(0.0,512.0);
	EggholderOptimizer test;


	test.setBounds(testBounds);

	test.setDimension(2);
	test.setInitialPopulationSize(500);
	test.setNumberOfNewIndividualsInAGeneration(500);
	test.setNumberOfDeathsInAGeneration(100);
	test.setMutationProbability(0.1);
	test.setNumberOfGenerations(50);
	test.setDisplayOff();
	test.setMaximumNumberOfGeneratedIndividuals(1000000);
	test.setNumberOfThreads(4);

	test.optimize();


	double objectiveFunctionValueOptimized = test.getBestObjectiveFunction();

	ASSERT_LT(objectiveFunctionValueOptimized, -500.0);


}




#endif
