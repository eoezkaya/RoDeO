/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
 *
 *  file is part of RoDeO
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */


#include "ea_optimizer.hpp"
#include "random_functions.hpp"
#include "auxiliary_functions.hpp"
#include "LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "LinearAlgebra/INCLUDE/vector_operations.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include<cassert>

using namespace arma;
using namespace std;


void EAOutput::printSolution(EAIndividual & input) const{

	if(ifScreenDisplay){

		input.printLess();

	}

}



EAIndividual::EAIndividual(unsigned int dim){

	assert(dim >0);
	dimension = dim;
	genes = zeros<vec>(dim);

}

EAIndividual::EAIndividual(){


}

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

	assert(value>0);
	assert(value<1.0);
	reproductionProbability = value;


}

void EAIndividual::setDeathProbabilityValue(double value){

	assert(value>=0);
	assert(value<1.0);
	deathProbability = value;


}


vec EAIndividual::getGenes(void) const{

	return genes;

}

void EAIndividual::setGenes(vec input){

	genes = input;
}

void EAIndividual::initializeGenes(vec values){

	assert(dimension == values.size());
	genes = values;

}


void EAIndividual::setObjectiveFunctionValue(double value){

	objectiveFunctionValue = value;

}

void EAIndividual::print(void) const{

	std::cout<<"\n";
	printScalar(id);
	trans(genes).print("x = ");
	printScalar(objectiveFunctionValue);
	printScalar(fitness);
	printScalar(reproductionProbability);
	printScalar(deathProbability);
	std::cout<<"\n";


}

void EAIndividual::printLess(void) const{

	std::cout<<"\n";
	trans(genes).print("x = ");
	printScalar(objectiveFunctionValue);
	std::cout<<"\n";

}

void EAIndividual::initializeRandom(void){

	assert(dimension>0);
	genes = generateRandomVector(0.0,1.0,dimension);
	fitness = generateRandomDouble(0.0,1.0);
	objectiveFunctionValue = generateRandomDouble(0.0,1.0);
	reproductionProbability = generateRandomDouble(0.0,0.1);
	deathProbability = generateRandomDouble(0.0,0.1);

}

void EAPopulation::setDimension(unsigned int value){

	dimension = value;

}

unsigned int EAPopulation::getSize(void) const{

	return population.size();

}

void EAPopulation::addIndividual(EAIndividual itemToAdd){

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

void EAPopulation::removeIndividual(unsigned int order){

	unsigned int id = getIdOftheIndividual(order);

	population.erase(population.begin() + order);

	if(id == idPopulationMaximum || id == idPopulationMinimum){

		updatePopulationMinAndMax();
	}
}

unsigned int EAPopulation::getIdOftheIndividual(unsigned int index) const{

	assert(index < getSize());
	return population[index].getId();

}

EAIndividual EAPopulation::getIndividual(unsigned int id) const{

	unsigned int order = getIndividualOrderInPopulationById(id);
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

void EAPopulation::print(void) const{

	std::ios oldState(nullptr);
	oldState.copyfmt(std::cout);
	std::setprecision(9);
	std::cout<<"\nEA Population has total "<< getSize() << " individuals ...\n";
	std::cout<<"fmin = "<<populationMinimum<<" fmax = "<<populationMaximum<<"\n";
	std::cout<<"fmin ID = "<<idPopulationMinimum<<" fmax ID = "<<idPopulationMaximum<<"\n";
	std::cout<<"ID    Obj. Fun.     Fitness        Rep. P.       Death P.\n";
	for(auto it = std::begin(population); it != std::end(population); ++it) {


		unsigned int id = it->getId();
		double J = it->getObjectiveFunctionValue();
		double fitness = it->getFitnessValue();
		double reproductionP = it->getReproductionProbability();
		double deathP = it->getDeathProbability();
		cout << scientific;
		std::cout << id << "   "<< std::showpos<< setw(6) << J <<"  "<< std::noshowpos<< setw(6) << fitness <<"  "<<setw(6) << reproductionP<<"  "<< setw(6) << deathP<<"\n";
	}

	std::cout.copyfmt(oldState);

}

void EAPopulation::reset(void){

	population.clear();

}

void EAPopulation::writeToFile(std::string filename) const{

	assert(isNotEmpty(filename));
	assert(dimension>0);
	unsigned int numberOfColumnsInTheDataBuffer = dimension + 1;
	unsigned int numberOfRowsInTheDataBuffer = getSize();

	mat dataBuffer(getSize(), numberOfColumnsInTheDataBuffer);


	unsigned int rowNumber = 0;
	for(auto it = std::begin(population); it != std::end(population); ++it) {

		vec buffer = it->getGenes();
		addOneElement<vec>(buffer,it->getObjectiveFunctionValue());
		rowvec rowBuffer = trans(buffer);
		dataBuffer.row(rowNumber) = rowBuffer;
		rowNumber++;
	}

	saveMatToCVSFile(dataBuffer,filename );

}

void EAPopulation::readFromFile(std::string filename){

	assert(isNotEmpty(filename));
	assert(dimension>0);
	reset();
	mat readBuffer = readMatFromCVSFile(filename);

	unsigned int numberOfIndividualsFoundInTheFile = readBuffer.n_rows;

	for(unsigned int i=0; i<numberOfIndividualsFoundInTheFile; i++){

		EAIndividual individualToAdd(dimension);

		rowvec rowBuffer = readBuffer.row(i);
		vec buffer = trans(rowBuffer);
		vec genes = buffer.head(dimension);
		individualToAdd.setGenes(genes);
		individualToAdd.setId(i);
		individualToAdd.setObjectiveFunctionValue(buffer(dimension));

		addIndividual(individualToAdd);

	}

	updatePopulationProperties();
	updatePopulationMinAndMax();
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

	populationMaximum = -LARGE;
	populationMinimum =  LARGE;
	for(auto it = std::begin(population); it != std::end(population); ++it) {


		double J = it->getObjectiveFunctionValue();
		unsigned int id = it->getId();

		updateMinAndMaxIfNecessary(J, id);
	}

}

void EAPopulation::updateFitnessValuesLinear() {
	for (auto it = std::begin(population); it != std::end(population); ++it) {
		double fitnessValue = (populationMaximum
				- it->getObjectiveFunctionValue())
																												/ (populationMaximum - populationMinimum);
		it->setFitnessValue(fitnessValue);
	}
}

void EAPopulation::updateFitnessValuesQuadratic() {
	vec coefficients =
			findPolynomialCoefficientsForQuadraticFitnessDistribution();
	for (auto it = std::begin(population); it != std::end(population); ++it) {
		double x = it->getObjectiveFunctionValue();
		double fitnessValue = coefficients(0) * x * x + coefficients(1) * x
				+ coefficients(2);
		it->setFitnessValue(fitnessValue);
	}
}

void EAPopulation::updateFitnessValues(void){

	/* we assume always minimization => maximum value has the worst fitness, minimum value has the best */

	if(polynomialForFitnessDistrubution == "linear"){

		updateFitnessValuesLinear();
	}

	if(polynomialForFitnessDistrubution == "quadratic"){

		updateFitnessValuesQuadratic();
	}

}

vec EAPopulation::findPolynomialCoefficientsForQuadraticFitnessDistribution(void) const{

	/* f(x) = ax^2 + bx + c
	 *
	 * f(populationMinimum)  = 1
	 * f(populationMaximum)  = aValueCloseButNotEqualToZero
	 * f'(populationMaximum) = 0
	 *
	 * */
	vec coefficients(3);
	mat A(3, 3, fill::zeros);
	double aValueCloseButNotEqualToZero = 0.001;

	A(0,0) = populationMinimum*populationMinimum;
	A(0,1) = populationMinimum;
	A(0,2) = 1.0;

	A(1,0) = populationMaximum*populationMaximum;
	A(1,1) = populationMaximum;
	A(1,2) = 1.0;

	A(2,0) = 2.0*populationMaximum;
	A(2,1) = 1.0;
	A(2,2) = 0.0;

	vec rhs(3);
	rhs(0) = 1.0 ;
	rhs(1) = aValueCloseButNotEqualToZero;
	rhs(2) = 0.0;

	coefficients = inv(A)*rhs;


	double f1 = coefficients(0)*populationMinimum*populationMinimum + coefficients(1) * populationMinimum + coefficients(2);
	assert(fabs(f1-1.0) < 10E-8);
	double f2 = coefficients(0)*populationMaximum*populationMaximum + coefficients(1) * populationMaximum + coefficients(2);
	assert(fabs(f2-aValueCloseButNotEqualToZero) < 10E-5);
	double fmin = 2*coefficients(0)*populationMaximum + coefficients(1);
	assert(fabs(fmin) < 10E-8);
	return coefficients;

}


void EAPopulation::updateReproductionProbabilities(void){

	double sumFitness = 0.0;

	for(auto it = std::begin(population); it != std::end(population); ++it) {

		sumFitness+=it->getFitnessValue();
	}

	assert(sumFitness > 0.0);


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

	vec P = { 0.9 };
	vec Q = quantile(fitnessValues, P);

	double best10PercentThreshold = Q(0);


	vec oneOverFitness(getSize());


	for(i=0; i<getSize(); i++) {

		if(fitnessValues(i) > best10PercentThreshold){

			oneOverFitness(i) = 0.0;
		}
		else{

			oneOverFitness(i) = 1.0/fitnessValues(i);

		}

	}

	double sumoneOverFitness = sum(oneOverFitness);

	i=0;
	for(auto it = std::begin(population); it != std::end(population); ++it) {


		double deathProbability = oneOverFitness(i)/sumoneOverFitness;
		it->setDeathProbabilityValue(deathProbability );
		i++;
	}


}


void EAPopulation::updatePopulationProperties(void){

	assert(getSize());
	updateFitnessValues();
	updateReproductionProbabilities();
	updateDeathProbabilities();

}


EAIndividual EAPopulation::pickUpARandomIndividualForReproduction(void) const{


	double probabilitySum = 0.0;
	double randomNumberBetweenOneAndZero = generateRandomDouble(0.0,1.0);
	for(auto it = std::begin(population); it != std::end(population); ++it) {

		probabilitySum += it->getReproductionProbability();


		if (randomNumberBetweenOneAndZero < probabilitySum) {

			return *it;
			break;
		}
	}

	assert(false);

}


unsigned int EAPopulation::pickUpAnIndividualThatWillDie(void) const{

	unsigned int order = 0;
	double probabilitySum = 0.0;
	double randomNumberBetweenOneAndZero = generateRandomDouble(0.0,1.0);
	for(auto it = std::begin(population); it != std::end(population); ++it) {

		probabilitySum += it->getDeathProbability();


		if (randomNumberBetweenOneAndZero < probabilitySum) {

			break;
		}

		order++;
	}

	return order;

}



EAOptimizer::EAOptimizer(){

	setNumberOfThreads(1);

}


void EAOptimizer::setDimension(unsigned int value){

	dimension = value;
	population.setDimension(value);
}


void EAOptimizer::setMutationProbability(double value){

	assert(value>0.0 && value <1.0);
	mutationProbability = value;
	mutationProbabilityLastGeneration = value;

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








void EAOptimizer::callObjectiveFunction(EAIndividual &individual){

	assert(dimension > 0);

	vec designVector = individual.getGenes();
	assert(designVector.size() == dimension);

	double objectiveFunctionValue = 0.0;

	if(ifObjectiveFunctionIsSet){
		objectiveFunctionValue = calculateObjectiveFunction(designVector);

	}

	else{

		objectiveFunctionValue = calculateObjectiveFunctionInternal(designVector);

	}

	individual.setObjectiveFunctionValue(objectiveFunctionValue);


}




EAIndividual  EAOptimizer::generateRandomIndividual(void){


//	this->output.printMessage("Random invidual = ");

	assert(parameterBounds.areBoundsSet());
	EAIndividual newIndividual(dimension);

	vec randomGenes = parameterBounds.generateVectorWithinBounds();
//	output.printMessage("x", randomGenes);
	newIndividual.initializeGenes(randomGenes);
	newIndividual.setId(totalNumberOfGeneratedIndividuals);
	totalNumberOfGeneratedIndividuals++;

	callObjectiveFunction(newIndividual);
//	output.printMessage("f(x) = ", newIndividual.getObjectiveFunctionValue());


	return newIndividual;

}


void EAOptimizer::initializePopulation(void){

	assert(sizeOfInitialPopulation>0);
	assert(areBoundsSet());


	output.printMessage("Size of initial population = ",sizeOfInitialPopulation);

#pragma omp parallel
	{

		std::vector<EAIndividual> slavePopulation;

#pragma omp for nowait
		for (unsigned int i = 0; i < sizeOfInitialPopulation; ++i)
		{

			if(i%100 == 0) {

				output.printMessage("Iteration = ",i);
			}
			EAIndividual generatedIndividual = generateRandomIndividual();
			slavePopulation.push_back(generatedIndividual);
		}


#pragma omp critical
		{
			/* merge populations from each thread */
			population.addAGroupOfIndividuals(slavePopulation);


		}
	}


	population.updatePopulationProperties();

	ifPopulationIsInitialized = true;
}


void EAOptimizer::printPopulation(void) const{

	population.print();

}

void EAOptimizer::printSettings(void) const{

	printScalar(dimension);
	printScalar(sizeOfInitialPopulation);
	printScalar(mutationProbability);
	printScalar(maxNumberOfFunctionEvaluations);
	printScalar(numberOfGenerations);
	printScalar(numberOfNewIndividualsInAGeneration);
	printScalar(numberOfDeathsInAGeneration);


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

std::pair<EAIndividual, EAIndividual> EAOptimizer::generateRandomParents(void) const{

	assert(ifPopulationIsInitialized);
	std::pair<EAIndividual, EAIndividual> parents;

	EAIndividual mother = population.pickUpARandomIndividualForReproduction();


	while (true) {

		EAIndividual father = population.pickUpARandomIndividualForReproduction();


		if (mother.getId() != father.getId()) {

			parents.first = mother;
			parents.second = father;
			break;
		}

	}

	return parents;
}


void EAOptimizer::applyMutation(vec& inputGenes) {


	assert(mutationProbability > 0.0 && mutationProbability < 1.0);
	assert(parameterBounds.areBoundsSet());

	vec randomGenes = parameterBounds.generateVectorWithinBounds();

	for(unsigned int i=0; i<inputGenes.size(); i++){

		double aRandomNumberBetweenZeroAndOne =  generateRandomDouble(0.0,1.0);

		if(aRandomNumberBetweenZeroAndOne < mutationProbability){

			totalNumberOfMutations++;
			inputGenes(i) = randomGenes(i);


		}


	}

}


void EAOptimizer::applyBoundsIfNecessary(EAIndividual &individual) const{

	assert(areBoundsSet());

	vec genes  = individual.getGenes();
	vec lb = parameterBounds.getLowerBounds();
	vec ub = parameterBounds.getUpperBounds();
	assert(genes.size() == lb.size());
	assert(genes.size() == ub.size());

	for(unsigned int i=0; i<lb.size(); i++){

		if(genes(i) < lb(i))  genes(i) = lb(i);
		if(genes(i) > ub(i))  genes(i) = ub(i);

	}

	individual.setGenes(genes);
}




EAIndividual EAOptimizer::crossOver(std::pair<EAIndividual, EAIndividual> parents){

	assert(parents.first.getId()  != parents.second.getId());

	vec motherGenes = parents.first.getGenes();
	vec fatherGenes = parents.second.getGenes();

	assert(motherGenes.size() == fatherGenes.size());

	vec newGenes = zeros<vec>(motherGenes.size());

	for(unsigned int i=0; i<motherGenes.size(); i++){

		newGenes(i) = generateRandomDoubleFromNormalDist(motherGenes(i), fatherGenes(i), 6.0);
		totalNumberOfCrossOvers++;

	}


	applyMutation(newGenes);


	EAIndividual child(dimension);
	child.setGenes(newGenes);
	child.setId(totalNumberOfGeneratedIndividuals);


	return child;

}


EAIndividual  EAOptimizer::generateIndividualByReproduction(std::pair<EAIndividual, EAIndividual> parents){

	assert(areBoundsSet());

	EAIndividual child =  crossOver(parents);
	applyBoundsIfNecessary(child);
	totalNumberOfGeneratedIndividuals++;

	return child;

}

void EAOptimizer::callObjectiveFunctionForAGroup(
		std::vector<EAIndividual> &children) {
#pragma omp parallel
	{
		for (auto it = std::begin(children); it != std::end(children); ++it) {
			callObjectiveFunction(*it);
		}
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
	callObjectiveFunctionForAGroup(children);

	population.addAGroupOfIndividuals(children);

}


void EAOptimizer::removeIndividualsFromPopulation(void) {


	for(unsigned int i=0; i<numberOfDeathsInAGeneration; i++){

		unsigned int order = population.pickUpAnIndividualThatWillDie();
		population.removeIndividual(order);
		population.updatePopulationProperties();

	}


}


void EAOptimizer::generateANewGeneration(void){

	output.printMessage("EA Optimizer: Generating a new generation");
	output.printMessage("EA Optimizer: Adding some individuals to the population");
	addNewIndividualsToPopulation();
	population.updatePopulationProperties();
	output.printMessage("EA Optimizer: Removing some individuals from the population");
	removeIndividualsFromPopulation();
	output.printMessage("EA Optimizer: Updating population properties");

	population.updatePopulationProperties();
	population.updatePopulationMinAndMax();


}


void EAOptimizer::checkIfSettingsAreOk(void) const{

	assert(areBoundsSet());
	assert(parameterBounds.checkIfBoundsAreValid());
	assert(numberOfNewIndividualsInAGeneration>0);
	assert(sizeOfInitialPopulation>0);
	assert(dimension >0);
	assert(numberOfGenerations >0);
	assert(maxNumberOfFunctionEvaluations>0);
	assert(numberOfDeathsInAGeneration < sizeOfInitialPopulation + numberOfNewIndividualsInAGeneration);

	if(ifWarmStart ){

		assert(ifFilenameWarmStartIsSet);
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
	output.printMessage("EA Optimizer: start...");
	checkIfSettingsAreOk();

	if(ifWarmStart){

#pragma omp critical
		{
			readWarmRestartFile();
		}


	}
	else{
		output.printMessage("EA Optimizer: initializing population...");
		initializePopulation();
	}

#if 0
	printPopulation();
#endif

	for(unsigned int i=0; i<numberOfGenerations; i++){

		output.printMessage("EA Optimizer: Generation = ",i);

		generateANewGeneration();

		if(totalNumberOfGeneratedIndividuals >= maxNumberOfFunctionEvaluations) break;


		printSolution();

	}

	output.printMessage("EA Optimization has been terminated...\n");

	printSolution();

	double mutationCrossOverRatio = double(totalNumberOfMutations)/double(totalNumberOfCrossOvers);
	output.printMessage("EA Optimization: total number of function evaluations = ", totalNumberOfGeneratedIndividuals);
	output.printMessage("EA Optimization: Effective mutation ratio = ",mutationCrossOverRatio);

}


void EAOptimizer::printSolution(void){

	EAIndividual solution = population.getTheBestIndividual();

	output.printMessage("The Optimal solution = ");
	output.printSolution(solution);


}

void EAOptimizer::setDisplayOn(void){

	output.ifScreenDisplay = true;

}


void EAOptimizer::writeWarmRestartFile(void){

	assert(ifFilenameWarmStartIsSet);
	output.printMessage("Writing the restart file ", filenameWarmStart);
	population.writeToFile(filenameWarmStart);

}

void EAOptimizer::readWarmRestartFile(void){

	assert(ifFilenameWarmStartIsSet);
	output.printMessage("Reading from the restart file ", filenameWarmStart);

	population.readFromFile(filenameWarmStart);
	ifPopulationIsInitialized = true;

}


void EAOptimizer::resetPopulation(void){

	population.reset();

}

