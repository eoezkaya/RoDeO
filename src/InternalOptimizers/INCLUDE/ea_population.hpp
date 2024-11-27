#ifndef EAPOPULATION_HPP
#define EAPOPULATION_HPP

#include "../../LinearAlgebra/INCLUDE/vector.hpp"
#include "ea_individual.hpp"
#include <limits>



namespace Rodop{

class EAPopulation{


private:

	static std::mt19937 gen;

	std::vector<EAIndividual> population;

	double populationMinimum = std::numeric_limits<double>::max();
	double populationMaximum = std::numeric_limits<double>::min();
	unsigned int idPopulationMinimum = 0;
	unsigned int idPopulationMaximum = 0;
	unsigned int dimension = 0;

	void updateMinAndMaxIfNecessary(double J, unsigned int id);
	std::string polynomialForFitnessDistrubution = "quadratic";

	vec findPolynomialCoefficientsForQuadraticFitnessDistribution(void) const;

	void updateFitnessValuesLinear();
	void updateFitnessValuesQuadratic();

	void checkIfAllObjectiveFunctionValuesAreZero() const;
	void checkIfAllFitnessValuesAreZero() const;



public:

	bool ifDisplay = false;

	double calculateSumOfDeathProbabilities() const;
	void writeTableAsCSV(const std::vector<std::vector<double>>& data, const std::string& filename) const;

	void writeToCSV(const std::string& filename) const;
	void readFromCSV(const std::string& filename);

	void setDimension(unsigned int);
	unsigned int getSize(void) const;

	void addIndividual(EAIndividual);
	void addAGroupOfIndividuals(std::vector<EAIndividual> individualsToAdd);

	void removeIndividual(unsigned int id);
	void removeIndividualById(unsigned int id);

	unsigned int getIdOftheIndividual(unsigned int index) const;

	EAIndividual getTheBestIndividual(void) const;
	EAIndividual getTheWorstIndividual(void) const;
	EAIndividual getIndividual(unsigned int) const;


	int getIndividualOrderInPopulationById(unsigned int id) const;
	unsigned int  getIndividualIdInPopulationByOrder(unsigned int order) const;

	void print(void) const;

	void updatePopulationMinAndMax(void);
	void updateFitnessValues(void);
	void updateReproductionProbabilities(void);
	void updateDeathProbabilities(void);
	void updatePopulationProperties(void);

	EAIndividual pickUpARandomIndividualForReproduction(void) const;
	EAIndividual pickUpAnIndividualThatWillDie(void) const;
	std::pair<EAIndividual, EAIndividual> generateRandomParents(void) const;

	void reset(void);
	bool hasRepeatedIDs() const;

};


} /*Namespace Rodop */

#endif
