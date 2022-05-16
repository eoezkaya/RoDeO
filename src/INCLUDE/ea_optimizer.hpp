/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
#ifndef EAOPTIMIZER_HPP
#define EAOPTIMIZER_HPP


#include<armadillo>
#include "bounds.hpp"
#include "output.hpp"
using namespace arma;

typedef double (*EAObjectiveFunction)(vec);


class EAIndividual {


private:

	unsigned int id = 0;
	double fitness = 0;
	double objectiveFunctionValue = 0;
	double reproductionProbability = 0;
	double deathProbability = 0;

	unsigned int dimension = 0;
	vec genes;

public:


	void setDimension(unsigned int);
	unsigned int getDimension(void) const;
	unsigned int getId(void) const;
	void setId(unsigned int);
	vec getGenes(void) const;
	void setGenes(vec);

	double getObjectiveFunctionValue(void) const;
	double getFitnessValue(void) const;
	double getReproductionProbability(void) const;
	double getDeathProbability(void) const;

	void setFitnessValue(double);
	void setObjectiveFunctionValue(double);
	void setReproductionProbabilityValue(double);
	void setDeathProbabilityValue(double);

	void initializeGenes(vec);

	void print(void) const;
	EAIndividual(unsigned int);


} ;




class EAOptimizer{

private:

	unsigned int dimension = 0;
	std::string optimizationType = "minimization";
	std::string polynomialForFitnessDistrubution = "quadratic";

	double populationMinimum = 0.0;
	double populationMaximum = 0.0;



	double mutationProbability = 0.0;
	double mutationProbabilityLastGeneration = 0.0;

	double (*calculateObjectiveFunction)(vec);

	Bounds parameterBounds;

	std::vector<EAIndividual> population;

	unsigned int idPopulationBest = 0;

	unsigned int sizeOfInitialPopulation = 0;
	unsigned int sizeOfPopulation = 0;
	unsigned int totalNumberOfGeneratedIndividuals = 0;
	unsigned int totalNumberOfGeneratedInviduals = 0;
	unsigned int numberOfNewIndividualsInAGeneration = 0;
	unsigned int numberOfDeathsInAGeneration = 0;
	unsigned int numberOfGenerations = 0;
	unsigned int maximumNumberOfGeneratedIndividuals = 0;
	unsigned int totalNumberOfMutations = 0;
	unsigned int totalNumberOfCrossOvers = 0;

	vec findPolynomialCoefficientsForQuadraticFitnessDistribution(void) const;
	vec crossOverGenes(unsigned int motherId, unsigned int fatherId);
	void applyBounds(vec& inputGenes) const;
	void addNewIndividualsToPopulation();

	void removeIndividualFromPopulation(unsigned int id);

	void addAGroupOfIndividualsToPopulation(std::vector<EAIndividual> individualsToAdd);
	void addIndividualToPopulation(EAIndividual individualToAdd);
	void removeIndividualsFromPopulation(void);

	unsigned int getIdOftheIndividual(unsigned int index) const;
	void findTheBestIndividualInPopulation(void);
	void printTheBestIndividual(void) const;


	bool ifPopulationIsInitialized = false;

	double improvementFunction = 0.0;

	OutputDevice output;

	virtual double calculateObjectiveFunctionInternal(vec);
	void addToList(std::vector<EAIndividual> &slavePopulation,
			std::vector<EAIndividual> &groupOfNewIndividualsToAdd);

public:

	bool ifObjectiveFunctionIsSet = false;


	EAOptimizer();

	void setDimension(unsigned int dim);
	unsigned int getDimension(void) const;

	bool isOptimizationTypeMinimization(void) const;
	bool isOptimizationTypeMaximization(void) const;
	bool areBoundsSet(void) const;

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setOptimizationType(std::string);
	void setBounds(Bounds);
	void setMutationProbability(double value);
	void setInitialPopulationSize(unsigned int);
	void setNumberOfNewIndividualsInAGeneration(unsigned int);
	void setNumberOfDeathsInAGeneration(unsigned int);
	void setNumberOfGenerations(unsigned int number);
	void setMaximumNumberOfGeneratedIndividuals(unsigned int number);
	void setNumberOfThreads(unsigned int nTreads);


	void setObjectiveFunction(EAObjectiveFunction);
	void callObjectiveFunction(EAIndividual &);



	unsigned int  getIndividualLocation(unsigned int id) const;
	unsigned int getNumberOfThreads(void) const;

	EAIndividual generateRandomIndividual(void);
	EAIndividual generateIndividualByReproduction(std::pair<unsigned int, unsigned int> indicesParents);

	void initializePopulation(void);
	void printPopulation(void) const;

	unsigned int getPopulationSize(void) const;

	vec getPopulationFitnessValues(void) const;
	vec getPopulationObjectiveFunctionValues(void) const;
	vec getPopulationReproductionProbabilities(void) const;
	vec getPopulationDeathProbabilities(void) const;
	void updatePopulationMinAndMax(void);


	void updatePopulationProperties(void);
	void updatePopulationFitnessValues(void);
	void updatePopulationReproductionProbabilities(void);
	void updatePopulationDeathProbabilities(void);

	unsigned int pickUpARandomIndividual(void) const;
	unsigned int pickUpAnIndividualThatWillDie() const;

	std::pair<unsigned int, unsigned int> generateRandomParents(void) const;
	void generateNewGeneration(void);

	void applyMutation(vec& inputGenes);

	void checkIfSettingsAreOk(void) const;

	void optimize(void);

	vec getBestDesignvector(void) const;

	double getBestObjectiveFunction(void) const;

};



#endif
