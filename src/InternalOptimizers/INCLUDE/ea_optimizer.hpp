#ifndef EAOPTIMIZER_HPP
#define EAOPTIMIZER_HPP

#include "./ea_individual.hpp"
#include "./ea_population.hpp"
#include "./general_purpose_optimizer.hpp"
#include "../../INCLUDE/globals.hpp"
#include<atomic>

#ifdef UNIT_TESTS
#include "gtest/gtest.h"
#endif





namespace Rodop{

typedef double (*EAObjectiveFunction)(vec);




class EAOptimizer : public GeneralPurposeOptimizer{

#ifdef UNIT_TESTS
	friend class EAOptimizerGeneralTest;
	FRIEND_TEST(EAOptimizerGeneralTest, CallObjectiveFunction);
	FRIEND_TEST(EAOptimizerGeneralTest, CallObjectiveFunctionForAGroup);
	FRIEND_TEST(EAOptimizerGeneralTest, CallObjectiveFunctionForAGroupMultiThread);
	FRIEND_TEST(EAOptimizerGeneralTest, CallObjectiveFunctionForAGroupMultiThreadSpeedTest);
	FRIEND_TEST(EAOptimizerGeneralTest, GenerateAGroupOfIndividualsForReproduction);
#endif


private:

	std::mt19937 gen;

	EAPopulation population;
	EAPopulation generated;
	EAPopulation removed;
	EAPopulation history;

	double sigma_factor = 6.0;

	double mutationProbability = 0.0;


	unsigned int sizeOfInitialPopulation = 0;

	unsigned int totalNumberOfGeneratedIndividuals = 0;
	unsigned int totalNumberOfCrossOvers = 0;


	unsigned int numberOfNewIndividualsInAGeneration = 0;
	unsigned int numberOfDeathsInAGeneration = 0;
	unsigned int numberOfGenerations = 0;
	unsigned int totalNumberOfMutations = 0;



	bool ifPopulationIsInitialized = false;

	double improvementFunction = 0.0;

	void callObjectiveFunctionForAGroup(std::vector<EAIndividual> &children);
	void generateAGroupOfIndividualsForReproduction(
			std::vector<EAIndividual> &children);
	void checkIfSettingsAreOk(void) const;

public:

	bool ifDisplay = false;

	EAOptimizer();

	void setMutationProbability(double value);
	void setInitialPopulationSize(unsigned int);
	void setNumberOfNewIndividualsInAGeneration(unsigned int);
	void setNumberOfDeathsInAGeneration(unsigned int);
	void setNumberOfGenerations(unsigned int number);
	void setMaximumNumberOfGeneratedIndividuals(unsigned int number);

	void callObjectiveFunction(EAIndividual &individual);


	EAIndividual generateRandomIndividual(void);

	void initializePopulation(void);

	void printPopulation(void) const;
	void printSettings() const;

	unsigned int getPopulationSize(void) const;

	std::pair<EAIndividual, EAIndividual> generateRandomParents(void) const;

	EAIndividual getSolution(void) const;
	vec getBestDesignVector(void) const;
	double getBestObjectiveFunctionValue(void) const;

	void applyMutation(vec& inputGenes);

	vec crossOver(const std::pair<EAIndividual, EAIndividual>& parents);

	EAIndividual generateIndividualByReproduction(std::pair<EAIndividual, EAIndividual> parents);
	void addNewIndividualsToPopulation(void);
	void removeIndividualsFromPopulation(void);
	void generateANewGeneration(void);

	void printSolution(void);
	void optimize(void);

	void setDimension(unsigned int value);
	void setWarmStartOn(void);
	void setWarmStartOff(void);

	void resetPopulation(void);
	void writeWarmRestartFile(void);
	void readWarmRestartFile(void);

	double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor);

	void writeHistoryPopulationToCSV(std::string filename);


};

} /* Namespace Rodop */

#endif
