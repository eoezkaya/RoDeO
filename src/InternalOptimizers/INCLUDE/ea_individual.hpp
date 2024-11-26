#ifndef EAINDIVIDUAL_HPP
#define EAINDIVIDUAL_HPP

#include "../../LinearAlgebra/INCLUDE/vector.hpp"
#include<random>



namespace Rodop{

class EAIndividual {


private:


	unsigned int id = 0;
	double fitness = 0;
	double objectiveFunctionValue = 0;
	double reproductionProbability = 0;
	double deathProbability = 0;
	unsigned int dimension = 0;
	vec genes;

	static std::mt19937 rng;

public:


	void setDimension(unsigned int);
	unsigned int getDimension(void) const;
	unsigned int getId(void) const;
	void setId(unsigned int);
	vec getGenes(void) const;
	void setGenes(const vec&);

	double getObjectiveFunctionValue(void) const;
	double getFitnessValue(void) const;
	double getReproductionProbability(void) const;
	double getDeathProbability(void) const;

	void setFitnessValue(double);
	void setObjectiveFunctionValue(double);
	void setReproductionProbabilityValue(double);
	void setDeathProbabilityValue(double);

	void initializeGenes(const vec&);
	void initializeRandom(void);

	void initializeVectorRandomBetweenZeroAndOne(vec& v);

	void print(void) const;
	void printLess(void) const;
	EAIndividual(unsigned int);
	EAIndividual();


};


} /*Namespace Rodop */

#endif
