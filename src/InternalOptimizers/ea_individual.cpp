#include "./INCLUDE/ea_individual.hpp"
#include<iostream>
#include<random>
#include<chrono>
#include <cassert>

using namespace std;


namespace Rodop{

std::mt19937 EAIndividual::rng(std::chrono::system_clock::now().time_since_epoch().count());


EAIndividual::EAIndividual(){}


EAIndividual::EAIndividual(unsigned int dim) : dimension(dim), genes(dim, 0.0) {}



unsigned int EAIndividual::getId(void) const{

	return id;

}


void  EAIndividual::setId(unsigned int idGiven){
	id = idGiven;
}

unsigned int EAIndividual::getDimension(void) const{

	return dimension;

}

double EAIndividual::getObjectiveFunctionValue(void) const{

	return objectiveFunctionValue;

}

double EAIndividual::getFitnessValue(void) const{

	return fitness;

}

double EAIndividual::getReproductionProbability(void) const{

	return reproductionProbability;

}

double EAIndividual::getDeathProbability(void) const{

	return deathProbability;

}


void EAIndividual::setFitnessValue(double value){

	fitness = value;

}

void EAIndividual::setReproductionProbabilityValue(double value){

	if (value <= 0) {

		std::cerr<<"EAIndividual::setReproductionProbabilityValue\n";
		std::cerr<<"Value = " << value << "\n";
		throw std::out_of_range("Value must be greater than 0.");
	}
	if (value >= 1.0) {
		std::cout << "Reproduction probability = " << value << "\n";
		throw std::out_of_range("Value must be less than 1.0.");
	}
	reproductionProbability = value;
}

void EAIndividual::setDeathProbabilityValue(double value){

	if (value < 0) {
		throw std::out_of_range("Value must be greater or equal to 0.");
	}
	if (value > 1.0) {
		std::cout << "Death probability = " << value << "\n";
		throw std::out_of_range("Value must be less than or equal to  1.0.");
	}
	deathProbability = value;
}


vec EAIndividual::getGenes(void) const{
	return genes;
}

void EAIndividual::setGenes(const vec& input){
    genes = input;
}

void EAIndividual::initializeGenes(const vec& values){
    if (dimension != values.getSize()) {
        throw std::invalid_argument("Size of input vector does not match the dimension.");
    }
    genes = values;
}



void EAIndividual::setObjectiveFunctionValue(double value){

	objectiveFunctionValue = value;

}

void EAIndividual::print(void) const{

	std::cout<<"\n";
	std::cout<<"ID = " << id << "\n";
	genes.print("x = ");

	std::cout<<"Objective function value = " << objectiveFunctionValue << "\n";
	std::cout<<"Fitness = "<< fitness << "\n";
	std::cout<<"Reproduction probability = " << reproductionProbability << "\n";
	std::cout<<"Death probability = " << deathProbability << "\n";
	std::cout<<"\n";


}

void EAIndividual::printLess(void) const{

	std::cout<<"\n";
	std::cout << "x: ";
	genes.print("x = \n");
	std::cout<<"Objective function value = " << objectiveFunctionValue << "\n";
	std::cout<<"\n";

}


void EAIndividual::initializeVectorRandomBetweenZeroAndOne(vec& v) {
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Range [0.0, 1.0)
    for(unsigned int i=0; i<v.getSize(); i++){
        v(i) = dist(rng);
    }
}


void EAIndividual::initializeRandom(void) {
    if (dimension <= 0) {
        throw std::invalid_argument("Dimension must be greater than zero.");
    }

    initializeVectorRandomBetweenZeroAndOne(genes);

    std::uniform_real_distribution<> dist01(0.0, 1.0);
    std::uniform_real_distribution<> dist010(0.0, 0.1);

    fitness = dist01(rng);
    objectiveFunctionValue = dist01(rng);
    reproductionProbability = dist010(rng);
    deathProbability = dist010(rng);
}


} /*Namespace Rodop */
