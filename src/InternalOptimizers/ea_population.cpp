#include "./INCLUDE/ea_population.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"
#include "../LinearAlgebra/INCLUDE/matrix.hpp"
#include <cassert>
#include<chrono>
#include<iostream>
#include<fstream>
#include <iomanip>
#include<limits>
#include <algorithm>
#include <set>

using namespace std;
namespace Rodop{

std::mt19937 EAPopulation::gen(std::chrono::system_clock::now().time_since_epoch().count());

void EAPopulation::setDimension(unsigned int value){

	dimension = value;

}

unsigned int EAPopulation::getSize(void) const{

	return population.size();

}

void EAPopulation::addIndividual(EAIndividual itemToAdd){

	if (itemToAdd.getDimension() != dimension) {
		throw std::invalid_argument("The dimension of the individual does not match the population dimension.");
	}

	population.push_back(itemToAdd);

	double J = itemToAdd.getObjectiveFunctionValue();
	unsigned int id = itemToAdd.getId();

	updateMinAndMaxIfNecessary(J, id);

}


void EAPopulation::addAGroupOfIndividuals(std::vector<EAIndividual> individualsToAdd){

	for(auto it = std::begin(individualsToAdd); it != std::end(individualsToAdd); ++it) {

		addIndividual(*it);

	}

}


void EAPopulation::removeIndividualById(unsigned int id) {

	unsigned int order = getIndividualOrderInPopulationById(id);

	removeIndividual(order);

}

void EAPopulation::removeIndividual(unsigned int order) {
    if (order >= population.size()) {
        throw std::out_of_range("Order index is out of bounds.");
    }

    unsigned int id = getIdOftheIndividual(order);

    if(ifDisplay){

    	std::cout<<"EA Popoulation: Removing the id = " << id << "\n";
    }
    population.erase(population.begin() + order);

    if (id == idPopulationMaximum || id == idPopulationMinimum) {
        updatePopulationMinAndMax();
    }
}


unsigned int EAPopulation::getIdOftheIndividual(unsigned int index) const{

	assert(index < getSize());
	return population[index].getId();

}

EAIndividual EAPopulation::getIndividual(unsigned int id) const{

	int order = getIndividualOrderInPopulationById(id);
	assert(order!=-1);
	return population.at(order);


}

EAIndividual EAPopulation::getTheBestIndividual(void) const{

	return getIndividual(idPopulationMinimum);
}

EAIndividual EAPopulation::getTheWorstIndividual(void) const{

	return getIndividual(idPopulationMaximum);
}

int  EAPopulation::getIndividualOrderInPopulationById(unsigned int id) const{


	unsigned int index = 0;
	for(auto it = std::begin(population); it != std::end(population); ++it) {

		if(id == it->getId()){

			return index;
		}

		index++;
	}

	/* if the id coud not be found return -1 as failure */
	return -1;
}

unsigned int  EAPopulation::getIndividualIdInPopulationByOrder(unsigned int order) const{

	assert(order < getSize());
	return population[order].getId();

}

void EAPopulation::print(void) const {
	// Save the current format of std::cout
	std::ios oldState(nullptr);
	oldState.copyfmt(std::cout);

	// Set the desired formatting
	std::cout << std::fixed << std::setprecision(6);

	std::cout << "\nEA Population has total " << getSize() << " individuals ...\n";
	std::cout << "fmin = " << populationMinimum << " fmax = " << populationMaximum << "\n";
	std::cout << "fmin ID = " << idPopulationMinimum << " fmax ID = " << idPopulationMaximum << "\n";
	std::cout << "-----------------------------------------------------------\n";
	std::cout << std::left << std::setw(8) << "ID"
			<< std::setw(15) << "Obj. Fun."
			<< std::setw(15) << "Fitness"
			<< std::setw(15) << "Rep. P."
			<< std::setw(15) << "Death P.\n";
	std::cout << "-----------------------------------------------------------\n";

	for (const auto& individual : population) {
		unsigned int id = individual.getId();
		double J = individual.getObjectiveFunctionValue();
		double fitness = individual.getFitnessValue();
		double reproductionP = individual.getReproductionProbability();
		double deathP = individual.getDeathProbability();

		std::cout << std::left << std::setw(8) << id
				<< std::setw(15) << J
				<< std::setw(15) << fitness
				<< std::setw(15) << reproductionP
				<< std::setw(15) << deathP << "\n";
	}

	std::cout << "-----------------------------------------------------------\n";

	// Restore the original formatting
	std::cout.copyfmt(oldState);
}

void EAPopulation::reset(void){

	population.clear();

}


void EAPopulation::writeTableAsCSV(const std::vector<std::vector<double>>& data, const std::string& filename) const {
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	for (const auto& row : data) {
		for (size_t i = 0; i < row.size(); ++i) {
			outFile << row[i];
			if (i < row.size() - 1) {
				outFile << ",";
			}
		}
		outFile << "\n";
	}

	outFile.close();
}


void EAPopulation::writeToCSV(const std::string& filename) const {
	if (filename.empty()) {
		throw std::invalid_argument("Filename cannot be empty.");
	}

	if (dimension <= 0) {
		throw std::invalid_argument("Dimension must be greater than 0.");
	}

	// Open file for writing
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	// Write the header
	outFile << "ID,ObjectiveFunctionValue,Fitness,ReproductionProbability,DeathProbability";
	for (unsigned int i = 0; i < dimension; ++i) {
		outFile << ",Gene_" << i;
	}
	outFile << "\n";

	// Write each individual's data to the file
	for (const auto& individual : population) {
		outFile << individual.getId() << ",";
		outFile << std::setprecision(9) << std::scientific << individual.getObjectiveFunctionValue() << ",";
		outFile << individual.getFitnessValue() << ",";
		outFile << individual.getReproductionProbability() << ",";
		outFile << individual.getDeathProbability();

		vec genes = individual.getGenes();
		for (unsigned int i = 0; i < genes.getSize(); ++i) {
			outFile << "," << genes(i);
		}
		outFile << "\n";
	}

	// Close the file
	outFile.close();
}

void EAPopulation::readFromCSV(const std::string& filename) {
	if (filename.empty()) {
		throw std::invalid_argument("Filename cannot be empty.");
	}

	std::ifstream inFile(filename);
	if (!inFile.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	population.clear();  // Clear any existing data

	std::string line;

	// Skip the header line
	if (!std::getline(inFile, line)) {
		throw std::runtime_error("Failed to read header line from CSV file");
	}

	while (std::getline(inFile, line)) {
		std::stringstream ss(line);
		std::string token;

		// Read ID
		std::getline(ss, token, ',');
		if (token.empty()) {
			throw std::runtime_error("Missing ID value in CSV");
		}
		unsigned int id = std::stoi(token);
		// Read objective function value
		std::getline(ss, token, ',');
		if (token.empty()) {
			throw std::runtime_error("Missing objective function value in CSV");
		}
		double objectiveFunctionValue = std::stod(token);

		// Read fitness value
		std::getline(ss, token, ',');
		if (token.empty()) {
			throw std::runtime_error("Missing fitness value in CSV");
		}
		double fitnessValue = std::stod(token);

		// Read reproduction probability
		std::getline(ss, token, ',');
		if (token.empty()) {
			throw std::runtime_error("Missing reproduction probability in CSV");
		}
		double reproductionProbability = std::stod(token);

		// Read death probability
		std::getline(ss, token, ',');
		if (token.empty()) {
			throw std::runtime_error("Missing death probability in CSV");
		}
		double deathProbability = std::stod(token);

		// Read genes
		std::vector<double> genes;
		for (unsigned int i = 0; i < dimension; ++i) {
			std::getline(ss, token, ',');
			if (token.empty()) {
				throw std::runtime_error("Missing gene value in CSV");
			}
			genes.push_back(std::stod(token));
		}


		// Create an individual and add to population
		EAIndividual individual(dimension);

		// Initialize the genes vector correctly
		vec geneVector(genes.size());
		for (size_t i = 0; i < genes.size(); ++i) {
			geneVector(i) = genes[i];
		}

		individual.setId(id);
		individual.initializeGenes(geneVector);
		individual.setObjectiveFunctionValue(objectiveFunctionValue);
		individual.setFitnessValue(fitnessValue);
		individual.setReproductionProbabilityValue(reproductionProbability);
		individual.setDeathProbabilityValue(deathProbability);

		addIndividual(individual);
	}

	inFile.close();

}



void EAPopulation::updateMinAndMaxIfNecessary(double J, unsigned int id) {

	if (J > populationMaximum) {
		populationMaximum = J;
		idPopulationMaximum = id;

	}
	if (J < populationMinimum) {
		populationMinimum = J;
		idPopulationMinimum = id;
	}
}

void EAPopulation::updatePopulationMinAndMax(void){

	populationMaximum =  std::numeric_limits<double>::min();;
	populationMinimum =  std::numeric_limits<double>::max();;
	for(auto it = std::begin(population); it != std::end(population); ++it) {


		double J = it->getObjectiveFunctionValue();
		unsigned int id = it->getId();

		updateMinAndMaxIfNecessary(J, id);
	}

}


void EAPopulation::checkIfAllObjectiveFunctionValuesAreZero() const{

	bool allZero = std::all_of(std::begin(population), std::end(population),
			[](const EAIndividual& individual) {
		return fabs(individual.getObjectiveFunctionValue()) < 1e-14;
	});

	if (allZero) {
		throw std::runtime_error("All objective function values are zero. Cannot update fitness values.");
	}


}


void EAPopulation::checkIfAllFitnessValuesAreZero() const{

	bool allZero = std::all_of(std::begin(population), std::end(population),
			[](const EAIndividual& individual) {
		return std::abs(individual.getFitnessValue()) < 1e-14;
	});

	if (allZero) {
		throw std::runtime_error("All fitness values are zero. Cannot update reproduction probablities.");
	}


}


void EAPopulation::updateFitnessValuesLinear() {

	checkIfAllObjectiveFunctionValuesAreZero();

	for (auto it = std::begin(population); it != std::end(population); ++it) {
		double fitnessValue = (populationMaximum
				- it->getObjectiveFunctionValue())
																																																/ (populationMaximum - populationMinimum);
		it->setFitnessValue(fitnessValue);
	}
}

void EAPopulation::updateFitnessValuesQuadratic() {

	checkIfAllObjectiveFunctionValuesAreZero();

	vec coefficients =
			findPolynomialCoefficientsForQuadraticFitnessDistribution();
	for (auto it = std::begin(population); it != std::end(population); ++it) {
		double x = it->getObjectiveFunctionValue();
		double fitnessValue = coefficients(0) * x * x + coefficients(1) * x
				+ coefficients(2);
		if(fitnessValue <= 0.0){
			print();
			std::cerr<< "WARNING: fitness value = " << fitnessValue << "\n";
			throw std::runtime_error("EAPopulation::updateFitnessValuesQuadratic: invalid fitness value");
		}


		it->setFitnessValue(fitnessValue);
	}
}

void EAPopulation::updateFitnessValues(void){

	if (population.size() == 0) {
		throw std::runtime_error("Population size must be greater than zero.");
	}

	/* we assume always minimization => maximum value has the worst fitness, minimum value has the best */

	if(polynomialForFitnessDistrubution == "linear"){

		updateFitnessValuesLinear();
	}

	if(polynomialForFitnessDistrubution == "quadratic"){

		updateFitnessValuesQuadratic();
	}

}


/* Here we assume a quadratic function between fitness and objective function values.
 *
 *
 * fitness(J) = a*x*x + b*x + c
 *
 * J(populationMaximum)  = aValueCloseButNotEqualToZero
 * J(populationMinimum)  = 1.0
 *
 */

vec EAPopulation::findPolynomialCoefficientsForQuadraticFitnessDistribution() const {
	vec coefficients(3);
	mat A(3,3);
	double aValueCloseButNotEqualToZero = 0.01;

	/*
	 * fitness = a*x*x + b*x + c
	 *
	 * fitness(min) = 1 (best fitness)
	 * fitness(max) = 0.001 (worst fitness)
	 * dfitness/dx = 2*a*x + b  = 0
	 *
	 * a * populationMinimum * populationMinimum + b*populationMinimum + c = 1
	 * a * populationMaximum * populationMaximum + b*populationMaximum + c = aValueCloseButNotEqualToZero
	 *
	 * derivative at populationMaximum = 0
	 *
	 *
	 * */

//	std::cout<<"populationMinimum = " << populationMinimum << "\n";
//	std::cout<<"populationMaximum = " << populationMaximum << "\n";

	A(0,0) = populationMinimum * populationMinimum;
	A(0,1) = populationMinimum;
	A(0,2) = 1.0;

	A(1,0) = populationMaximum * populationMaximum;
	A(1,1) = populationMaximum;
	A(1,2) = 1.0;

	A(2,0) = 2.0 * populationMaximum;
	A(2,1) = 1.0;
	A(2,2) = 0.0;

	vec rhs(3);
	rhs(0) = 1.0;
	rhs(1) = aValueCloseButNotEqualToZero;
	rhs(2) = 0.0;

	mat A_inv = A.invert();
	coefficients = A_inv.matVecProduct(rhs);

//	double f1 = coefficients(0)*populationMinimum * populationMinimum + coefficients(1) * populationMinimum + coefficients(2);
//	double f2 = coefficients(0)*populationMaximum * populationMaximum + coefficients(1) * populationMaximum + coefficients(2);
//	double f3 = 2*coefficients(0)*populationMaximum + coefficients(1);
//
//	std::cout<<"f1 = " << f1 << "\n";
//	std::cout<<"f2 = " << f2 << "\n";
//	std::cout<<"f3 = " << f3 << "\n";

	return coefficients;
}



void EAPopulation::updateReproductionProbabilities(void){

	double sumFitness = 0.0;

	for(auto it = std::begin(population); it != std::end(population); ++it) {

		sumFitness+=it->getFitnessValue();
	}

	for(auto it = std::begin(population); it != std::end(population); ++it) {

		double reproductionProbability = it->getFitnessValue()/sumFitness;
		it->setReproductionProbabilityValue(reproductionProbability);

	}


}


void EAPopulation::updateDeathProbabilities(void){

	vec fitnessValues(getSize());
	unsigned int i = 0;
	for(auto it = std::begin(population); it != std::end(population); ++it) {

		fitnessValues(i) = it->getFitnessValue();
		i++;
	}

	/* 90 percent of the values are below this threshold */
	double best10PercentThreshold = fitnessValues.quantile(0.9);

	//	std::cout<< " best10PercentThreshold:  " << best10PercentThreshold << "\n";

	vec oneOverFitness(getSize());


	for(i=0; i<getSize(); i++) {

		if(fitnessValues(i) > best10PercentThreshold){
			oneOverFitness(i) = 0.0;
		}
		else{
			oneOverFitness(i) = 1.0/fitnessValues(i);
		}
	}

	double sumoneOverFitness = oneOverFitness.sum();

	i=0;
	for(auto it = std::begin(population); it != std::end(population); ++it) {
		double deathProbability = oneOverFitness(i)/sumoneOverFitness;
		//		std::cout<<"deathProbability = " << deathProbability << "\n";
		it->setDeathProbabilityValue(deathProbability );
		i++;
	}
}


void EAPopulation::updatePopulationProperties(void){

	if (population.size() == 0) {
		throw std::runtime_error("Population size must be greater than zero.");
	}
//	std::cout<<"Updating fitness values...\n";
	updateFitnessValues();
//	std::cout<<"Updating reproduction probabilities...\n";
	updateReproductionProbabilities();
//	std::cout<<"Updating death probabilities...\n";
	updateDeathProbabilities();

}


EAIndividual EAPopulation::pickUpARandomIndividualForReproduction(void) const {
	double probabilitySum = 0.0;

	// Use the class's static random number generator
	std::uniform_real_distribution<> dist01(0.0, 1.0);
	double randomNumberBetweenOneAndZero = dist01(gen);

	for (auto it = std::begin(population); it != std::end(population); ++it) {
		probabilitySum += it->getReproductionProbability();

		if (randomNumberBetweenOneAndZero < probabilitySum) {
			return *it;
		}
	}

	// This point should not be reached if the logic is correct
	throw std::runtime_error("Failed to pick a random individual for reproduction.");
}

std::pair<EAIndividual, EAIndividual> EAPopulation::generateRandomParents() const {

    if (population.size() < 2) {
        throw std::runtime_error("Population size must have at least 2 members for reproduction.");
    }

    EAIndividual mother = pickUpARandomIndividualForReproduction();
    EAIndividual father;

    do {
        father = pickUpARandomIndividualForReproduction();
    } while (mother.getId() == father.getId());

    return std::make_pair(mother, father);
}


EAIndividual EAPopulation::pickUpAnIndividualThatWillDie(void) const {

	double probabilitySum = 0.0;

	// Use the class's static random number generator
	std::uniform_real_distribution<> dist01(0.0, 1.0);
	double randomNumberBetweenOneAndZero = dist01(gen);

	//    std::cout<<"random number = " << randomNumberBetweenOneAndZero << "\n";
	for (auto it = std::begin(population); it != std::end(population); ++it) {
		//   	std::cout<<" probabilitySum = " <<  probabilitySum << "\n";
		probabilitySum += it->getDeathProbability();

		if (randomNumberBetweenOneAndZero < probabilitySum) {
			return *it;
		}
	}


	throw std::runtime_error("Failed to pick a valid individual that will die.");

}

double EAPopulation::calculateSumOfDeathProbabilities() const {
	double sum = 0.0;
	for (const auto& individual : population) {
		sum += individual.getDeathProbability();
	}
	return sum;
}


bool EAPopulation::hasRepeatedIDs() const {
        std::set<unsigned int> idSet;  // Set to store unique IDs

        // Iterate through the population
        for (const auto& individual : population) {
            unsigned int id = individual.getId();

            // Check if the ID is already in the set
            if (idSet.find(id) != idSet.end()) {
                // ID already exists, so there is a duplicate
                return true;
            }

            // Add the ID to the set
            idSet.insert(id);
        }

        // No duplicates found
        return false;
    }


} /* Namespace Rodop */
