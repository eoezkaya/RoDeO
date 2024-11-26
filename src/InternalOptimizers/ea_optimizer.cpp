#include "./INCLUDE/ea_optimizer.hpp"
#include "./INCLUDE/ea_population.hpp"
#include "../LinearAlgebra/INCLUDE/matrix.hpp"

#include <iomanip>
#include<iostream>
#include<fstream>
#include <random>
#include<chrono>
#include <cassert>



namespace Rodop{


EAOptimizer::EAOptimizer(): gen(std::random_device{}()) {}



void EAOptimizer::setDimension(unsigned int value){

	dimension = value;
	population.setDimension(value);
	generated.setDimension(value);
	removed.setDimension(value);
	history.setDimension(value);
}


void EAOptimizer::setMutationProbability(double value){

	if (value <= 0.0 || value >= 1.0) {
		throw std::invalid_argument("Mutation probability must be between 0 and 1 (exclusive).");
	}
	mutationProbability = value;

}



void EAOptimizer::setInitialPopulationSize(unsigned int size){

	sizeOfInitialPopulation = size;

}


void EAOptimizer::setNumberOfNewIndividualsInAGeneration(unsigned int number){

	numberOfNewIndividualsInAGeneration = number;

}

void EAOptimizer::setNumberOfDeathsInAGeneration(unsigned int number){

	numberOfDeathsInAGeneration = number;

}

void EAOptimizer::setNumberOfGenerations(unsigned int number){

	numberOfGenerations = number;

}

void EAOptimizer::setMaximumNumberOfGeneratedIndividuals(unsigned int number){

	setMaxNumberOfFunctionEvaluations(number);

}

void EAOptimizer::callObjectiveFunction(EAIndividual &individual) {
	if (dimension == 0) {
		throw std::runtime_error("Error: Optimizer dimension must be set before calling the objective function.");
	}

	vec designVector = individual.getGenes();
	if (designVector.getSize() != dimension) {
		throw std::runtime_error("Error: The design vector size (" + std::to_string(designVector.getSize()) +
				") does not match the optimizer dimension (" + std::to_string(dimension) + ").");
	}

	double objectiveFunctionValue = 0.0;

	if (ifObjectiveFunctionIsSet) {
		objectiveFunctionValue = calculateObjectiveFunction(designVector.getPointer());
	} else {
		objectiveFunctionValue = calculateObjectiveFunctionInternal(designVector);
	}

	individual.setObjectiveFunctionValue(objectiveFunctionValue);
}


EAIndividual  EAOptimizer::generateRandomIndividual(void){


	if (!ifBoundsAreSet) {
		throw std::runtime_error("Bounds must be set before calling this method.");
	}

	EAIndividual newIndividual(dimension);
	vec randomGenes = vec::generateRandomVectorBetween(lowerBounds, upperBounds);

	//	output.printMessage("x", randomGenes);
	newIndividual.initializeGenes(randomGenes);
	unsigned int id = ++totalNumberOfGeneratedIndividuals;
	newIndividual.setId(id);


	callObjectiveFunction(newIndividual);
	//	output.printMessage("f(x) = ", newIndividual.getObjectiveFunctionValue());


	return newIndividual;

}


void EAOptimizer::initializePopulation(void) {

	population.reset();

	if (sizeOfInitialPopulation <= 0) {
		throw std::runtime_error("Error in initializePopulation: Initial population size must be greater than zero.");
	}
	if (!ifBoundsAreSet) {
		throw std::runtime_error("Error in initializePopulation: Bounds must be set before calling this method.");
	}

	// Temporary storage for the entire population
	std::vector<EAIndividual> totalPopulation;

	if(ifDisplay){
		std::cout<<"EAOPtimizer: Initializing population with " << sizeOfInitialPopulation << " individuals...\n";
	}


	for (unsigned int i = 0; i < sizeOfInitialPopulation; ++i) {
		EAIndividual generatedIndividual = generateRandomIndividual();
		totalPopulation.push_back(generatedIndividual);
	}

	// Add the total population to the main population
	population.addAGroupOfIndividuals(totalPopulation);
	history.addAGroupOfIndividuals(totalPopulation);
	population.updatePopulationProperties();
	ifPopulationIsInitialized = true;

	if(population.hasRepeatedIDs()){
		throw std::runtime_error("Error in initializePopulation: Initial population has repeated IDs.");

	}


}

void EAOptimizer::printPopulation(void) const{
	population.print();
}

void EAOptimizer::printSettings() const {
	std::cout << "=================================================" << std::endl;
	std::cout << "Evolutionary Algorithm Settings:" << std::endl;
	std::cout << "=================================================" << std::endl;

	std::cout << "Initial population Size: " << sizeOfInitialPopulation << std::endl;
	std::cout << "Mutation Rate: " << mutationProbability << std::endl;
	std::cout << "Maximum number of function evaluations: " << maxNumberOfFunctionEvaluations  << std::endl;
	std::cout << "Number of Generations: " << numberOfGenerations << std::endl;

	std::cout << "=================================================" << std::endl;
}

unsigned int EAOptimizer::getPopulationSize(void) const{

	return population.getSize();
}


EAIndividual EAOptimizer::getSolution(void) const{

	return population.getTheBestIndividual();

}

vec EAOptimizer::getBestDesignVector(void) const{

	EAIndividual best = getSolution();
	return best.getGenes();

}

double EAOptimizer::getBestObjectiveFunctionValue(void) const{

	EAIndividual best = getSolution();
	return best.getObjectiveFunctionValue();

}

std::pair<EAIndividual, EAIndividual> EAOptimizer::generateRandomParents() const {

	if (!ifPopulationIsInitialized) {
		throw std::runtime_error("Population must be initialized before generating random parents.");
	}
	return population.generateRandomParents();
}

void EAOptimizer::applyMutation(vec& inputGenes) {

	// Validate mutationProbability
	if (mutationProbability <= 0.0 || mutationProbability >= 1.0) {
		throw std::invalid_argument("Mutation probability must be between 0 and 1 (exclusive).");
	}

	// Validate that parameter bounds are set
	if (!ifBoundsAreSet) {
		throw std::runtime_error("Parameter bounds must be set before applying mutation.");
	}

	// Initialize random number generators
	std::uniform_real_distribution<> dist(0.0, 1.0);

	vec randomGenes = vec::generateRandomVectorBetween(lowerBounds, upperBounds);
	for (unsigned int i = 0; i < inputGenes.getSize(); ++i) {
		double randomValue = dist(gen);
		if (randomValue < mutationProbability) {
			inputGenes(i) = randomGenes(i);
			++totalNumberOfMutations;
		}
	}
}


double EAOptimizer::generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor) {
	double sigma = fabs((xe - xs)) / sigma_factor;
	double mu = (xe + xs) / 2.0;

	// Avoid zero sigma
	if (sigma == 0.0) sigma = 1.0;

	std::normal_distribution<double> distribution(mu, sigma);

	return distribution(gen);
}



vec EAOptimizer::crossOver(const std::pair<EAIndividual, EAIndividual>& parents) {
	// Check that parents are different
	if (parents.first.getId() == parents.second.getId()) {
		throw std::invalid_argument("Parent individuals must have different IDs.");
	}

	vec motherGenes = parents.first.getGenes();
	vec fatherGenes = parents.second.getGenes();

	// Check that gene sizes are equal
	if (motherGenes.getSize() != fatherGenes.getSize()) {
		throw std::invalid_argument("Mother and father genes must be of the same size.");
	}

	vec newGenes(motherGenes.getSize());

	// Generate new genes using a normal distribution-based crossover
	for (unsigned int i = 0; i < motherGenes.getSize(); ++i) {
		newGenes(i) = generateRandomDoubleFromNormalDist(motherGenes(i), fatherGenes(i), sigma_factor);
		++totalNumberOfCrossOvers;  // Thread-safe increment
	}

	// Apply mutation to the new genes
	applyMutation(newGenes);

	// Enforce the lower and upper bounds on the new genes
	for (unsigned int i = 0; i < dimension; ++i) {
		if (newGenes(i) < lowerBounds(i)) newGenes(i) = lowerBounds(i);
		if (newGenes(i) > upperBounds(i)) newGenes(i) = upperBounds(i);
	}

	return newGenes;
}


EAIndividual  EAOptimizer::generateIndividualByReproduction(std::pair<EAIndividual, EAIndividual> parents){

	if (!ifBoundsAreSet) {
		throw std::runtime_error("Bounds must be set before generating an individual by reproduction.");
	}

	EAIndividual child(dimension);
	vec newGenes =  crossOver(parents);

	child.setGenes(newGenes);
	child.setId(totalNumberOfGeneratedIndividuals);
	++totalNumberOfGeneratedIndividuals;
	return child;

}


void EAOptimizer::callObjectiveFunctionForAGroup(std::vector<EAIndividual> &children) {

	for (size_t i = 0; i < children.size(); ++i) {
		callObjectiveFunction(children[i]);
	}
}


void EAOptimizer::generateAGroupOfIndividualsForReproduction(
		std::vector<EAIndividual> &children) {
	for (unsigned int i = 0; i < numberOfNewIndividualsInAGeneration; ++i) {
		std::pair<EAIndividual, EAIndividual> parents = generateRandomParents();
		EAIndividual child = generateIndividualByReproduction(parents);
		children.push_back(child);
	}
}

void EAOptimizer::addNewIndividualsToPopulation(void) {

	assert(numberOfNewIndividualsInAGeneration > 0);


	std::vector<EAIndividual> children;
	generateAGroupOfIndividualsForReproduction(children);


	/* only this part can run parallel */
	callObjectiveFunctionForAGroup(children);
	/* only this part can run parallel */

	if(ifDisplay){
		std::cout<<"EA Optimizer: New individuals...\n";
		for (const auto& child : children) {
			child.print();
		}
	}
	population.addAGroupOfIndividuals(children);
	generated.addAGroupOfIndividuals(children);
	history.addAGroupOfIndividuals(children);

}


void EAOptimizer::removeIndividualsFromPopulation(void) {


	for(unsigned int i=0; i<numberOfDeathsInAGeneration; i++){

		EAIndividual individualToRemove = population.pickUpAnIndividualThatWillDie();

		removed.addIndividual(individualToRemove);

		if(ifDisplay){
			std::cout<< "EA Optimizer: Removing an individual...\n";
			individualToRemove.print();
		}

		population.removeIndividualById(individualToRemove.getId());
		population.updatePopulationProperties();

	}


}


void EAOptimizer::generateANewGeneration(void){

	if(ifDisplay){
		std::cout<<"EA Optimizer: Generating a new generation...\n";
	}
	addNewIndividualsToPopulation();
	population.updatePopulationProperties();

	if(ifDisplay){
		std::cout<<"EA Optimizer: Removing some individuals from the population...\n";
	}

	removeIndividualsFromPopulation();

	if(ifDisplay){
		std::cout<<"EA Optimizer: Updating population properties...\n";
	}

	population.updatePopulationProperties();
	population.updatePopulationMinAndMax();
}


void EAOptimizer::checkIfSettingsAreOk(void) const{

	if (!ifBoundsAreSet) {
		throw std::runtime_error("Bounds must be set before calling this method.");
	}

	if (numberOfNewIndividualsInAGeneration <= 0) {
		throw std::invalid_argument("Number of new individuals in a generation must be greater than 0.");
	}

	if (sizeOfInitialPopulation <= 0) {
		throw std::invalid_argument("Size of the initial population must be greater than 0.");
	}

	if (dimension <= 0) {
		throw std::invalid_argument("Dimension must be greater than 0.");
	}

	if (numberOfGenerations <= 0) {
		throw std::invalid_argument("Number of generations must be greater than 0.");
	}

	if (maxNumberOfFunctionEvaluations <= 0) {
		throw std::invalid_argument("Max number of function evaluations must be greater than 0.");
	}

	if (numberOfDeathsInAGeneration >= sizeOfInitialPopulation + numberOfNewIndividualsInAGeneration) {
		throw std::invalid_argument("Number of deaths in a generation must be less than the total population size.");
	}

	if(ifWarmStart ){

		if (!ifFilenameWarmStartIsSet) {
			throw std::runtime_error("Filename for warm start has not been set.");
		}

	}

}

void EAOptimizer::setWarmStartOn(void){

	ifWarmStart = true;
}
void EAOptimizer::setWarmStartOff(void){

	ifWarmStart = false;
}

void EAOptimizer::optimize(void){

	totalNumberOfGeneratedIndividuals = 0;
	//	output.printMessage("EA Optimizer: start...");
	checkIfSettingsAreOk();

	if(ifWarmStart){

		readWarmRestartFile();
	}
	else{
		//		output.printMessage("EA Optimizer: initializing population...");
		initializePopulation();

		if(ifDisplay){
			std::cout<<"EA Optimizer: Initial population = \n";
			printPopulation();
		}

	}

#if 0
	printPopulation();
#endif

	for(unsigned int i=0; i<numberOfGenerations; i++){

		if(ifDisplay){

			std::cout<<"EA Optimizer: Generation = " << i << "\n";
		}

		generateANewGeneration();

		if(totalNumberOfGeneratedIndividuals >= maxNumberOfFunctionEvaluations) break;

	}

	if(ifDisplay){

		std::cout<<"EA Optimization has been terminated...\n";
	}

	printSolution();

	double mutationCrossOverRatio = double(totalNumberOfMutations)/double(totalNumberOfCrossOvers);

	if(ifDisplay){

		std::cout<< "EA Optimization: Total number of function evaluations = " <<totalNumberOfGeneratedIndividuals << "\n";
		std::cout<< "EA Optimization: Effective mutation ratio = "<< mutationCrossOverRatio<< "\n";
	}

	if(ifSaveWarmStartFile){
		writeWarmRestartFile();
	}

	if(ifDisplay){
		std::cout<< "EA Optimization: removed individuals =\n";
		removed.print();
		std::cout<< "EA Optimization: generated individuals =\n";
		removed.print();
		std::cout<< "EA Optimization: remaining population =\n";
		printPopulation();
	}


}


void EAOptimizer::printSolution(void){

	if(ifDisplay){

		EAIndividual solution = population.getTheBestIndividual();
		std::cout<<"The optimal solution = \n";
		solution.printLess();

	}
	EAIndividual solution = population.getTheBestIndividual();

}



void EAOptimizer::writeWarmRestartFile() {
	// Check if the filename for warm start is set
	if (!ifFilenameWarmStartIsSet) {
		throw std::runtime_error("Filename for warm start is not set.");
	}

	// Optionally, you can print or log a message before writing
	// std::cout << "Writing the restart file: " << filenameWarmStart << std::endl;

	try {
		population.writeToCSV(filenameWarmStart);
	} catch (const std::exception& e) {
		// Handle exceptions from writeToCSV
		throw std::runtime_error("Failed to write warm restart file: " + std::string(e.what()));
	}
}

void EAOptimizer::readWarmRestartFile() {
	// Check if the filename for warm start is set
	if (!ifFilenameWarmStartIsSet) {
		throw std::runtime_error("Filename for warm start is not set.");
	}

	// Optionally, you can print or log a message before reading
	// std::cout << "Reading from the restart file: " << filenameWarmStart << std::endl;

	try {
		population.readFromCSV(filenameWarmStart);
		ifPopulationIsInitialized = true;  // Set the flag after successful read
	} catch (const std::exception& e) {
		// Handle exceptions from readFromCSV
		throw std::runtime_error("Failed to read warm restart file: " + std::string(e.what()));
	}
}

void EAOptimizer::resetPopulation(void){

	population.reset();

}


void EAOptimizer::writeHistoryPopulationToCSV(std::string filename){

	history.writeToCSV(filename);

}



} /* Namespace Rodop */
