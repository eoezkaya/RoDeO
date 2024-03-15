/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */


#include "./INCLUDE/optimization.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"

bool Optimizer::isConstrained(void) const{

	if(numberOfConstraints > 0) return true;
	else return false;
}

bool Optimizer::isNotConstrained(void) const{

	if(numberOfConstraints == 0) return true;
	else return false;
}

void Optimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;
	globalOptimalDesign.setNumberOfConstraints(numberOfConstraints);
	statistics.numberOfConstraintEvaluations.push_back(0);
	statistics.numberOfConstraintGradientEvaluations.push_back(0);
}



void Optimizer::estimateConstraints(DesignForBayesianOptimization &design) const{

	rowvec x = design.dv;
	assert(design.constraintValues.size() == numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::pair<double, double> result = it->interpolateWithVariance(x);

		design.constraintValues(it->getID()) = result.first;
		design.constraintSigmas(it->getID()) = result.second;

	}
}

void Optimizer::estimateConstraintsGradientStep(DesignForBayesianOptimization &design) const{

	rowvec x = design.dv;
	assert(design.constraintValues.size() == numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::pair<double, double> result = it->interpolateWithVariance(x);

		design.constraintValues(it->getID()) = it->interpolateUsingDerivatives(x);
		design.constraintSigmas(it->getID()) = result.second;

	}
}




bool Optimizer::checkConstraintFeasibility(rowvec constraintValues) const{

	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		bool ifFeasible = it->checkFeasibility(constraintValues(i));

		if(ifFeasible == false) {
			return false;
		}
		i++;
	}

	return true;
}



void Optimizer::calculateFeasibilityProbabilities(DesignForBayesianOptimization &designCalculated) const{

	rowvec probabilities(numberOfConstraints, fill::zeros);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		string type  = it->getInequalityType();
		double inequalityValue = it->getInequalityTargetValue();
		int ID = it->getID();
		double estimated = designCalculated.constraintValues(ID);
		double sigma = designCalculated.constraintSigmas(ID);

		if(type.compare(">") == 0){
			/* p (constraint value > target) */
			probabilities(ID) =  calculateProbalityGreaterThanAValue(inequalityValue, estimated, sigma);
		}

		if(type.compare("<") == 0){
			probabilities(ID) =  calculateProbalityLessThanAValue(inequalityValue, estimated, sigma);
		}

	}

	designCalculated.constraintFeasibilityProbabilities = probabilities;

}


void Optimizer::evaluateConstraints(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->evaluateDesign(d);
		unsigned int ID = it->getID();
		statistics.numberOfConstraintEvaluations[ID]++;

	}
}


void Optimizer::trainSurrogatesForConstraints() {
	output.printMessage("Training surrogate model for the constraints...");
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end();
			it++) {
		it->trainSurrogate();
	}
	output.printMessage("Model training for constraints is done...");
}

void Optimizer::computeConstraintsandPenaltyTerm(Design &d) {

	if(isConstrained()){

		output.printMessage("Evaluating constraints...");

		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);
		if(!ifConstraintsSatisfied){

			output.printMessage("The new sample does not satisfy all the constraints");
			d.isDesignFeasible = false;

		}
		else{

			output.printMessage("The new sample satisfies all the constraints");
			d.isDesignFeasible = true;
		}

		output.printMessage("Evaluation of the constraints is ready...");
	}

}

void Optimizer::addConstraintValuesToData(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->addDesignToData(d);
	}

}

void Optimizer::printConstraints(void) const{

	std::cout<< "List of constraints = \n";

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->print();
	}

}

