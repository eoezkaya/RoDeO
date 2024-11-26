#include "./INCLUDE/optimization.hpp"
#include "../LinearAlgebra/INCLUDE/vector.hpp"
#include<cassert>


namespace Rodop{


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

	vec x = design.dv;
	assert(design.constraintValues.getSize() == numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::pair<double, double> result;
		if(it->isUserDefinedFunction()){

			vec xNotNormalized = x.denormalizeVector(lowerBounds, upperBounds);
			result = it->interpolateWithVariance(xNotNormalized);
		}
		else{
			result = it->interpolateWithVariance(x);
		}

		design.constraintValues(it->getID()) = result.first;
		design.constraintSigmas(it->getID()) = result.second;

	}
}

void Optimizer::estimateConstraintsGradientStep(DesignForBayesianOptimization &design) const{

	vec x = design.dv;
	assert(design.constraintValues.getSize() == numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::pair<double, double> result = it->interpolateWithVariance(x);

		design.constraintValues(it->getID()) = it->interpolateUsingDerivatives(x);
		design.constraintSigmas(it->getID()) = result.second;

	}
}

bool Optimizer::checkConstraintFeasibility(const vec& constraintValues) const {
    if (constraintValues.getSize() < constraintFunctions.size()) {
        throw std::invalid_argument("Insufficient constraint values for feasibility check.");
    }

    unsigned int i = 0;
    for (const auto& constraintFunction : constraintFunctions) {
        if (!constraintFunction.checkFeasibility(constraintValues(i))) {
        	printInfoToLog("Constraint " + std::to_string(i) + " is not satisfied");
        	printInfoToLog("Constraint value = ", constraintValues(i));

            return false;  // Return immediately if any constraint is infeasible
        }
        else{
        	printInfoToLog("Constraint " + std::to_string(i) + " is satisfied");
        	printInfoToLog("Constraint value = ", constraintValues(i));

        }
        i++;
    }

    return true;  // All constraints are feasible
}


void Optimizer::calculateFeasibilityProbabilities(DesignForBayesianOptimization &designCalculated) const{

	vec probabilities(numberOfConstraints);

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		string type  = it->getInequalityType();
		double inequalityValue = it->getInequalityTargetValue();
		int ID = it->getID();
		double estimated = designCalculated.constraintValues(ID);
		double sigma = designCalculated.constraintSigmas(ID);

		if(type.compare(">") == 0){
			/* p (constraint value > target) */
			probabilities(ID) =  calculateProbabilityGreaterThanAValue(inequalityValue, estimated, sigma);
		}

		if(type.compare("<") == 0){
			probabilities(ID) =  calculateProbabilityLessThanAValue(inequalityValue, estimated, sigma);
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
    printInfoToLog("Training surrogate models for the constraints...");

    int index = 0;
    for (auto& constraint : constraintFunctions) {
        printInfoToLog("Training surrogate model for constraint #" + std::to_string(index) + "...");

        // Train the surrogate model
        try {
            constraint.trainSurrogate();
            printInfoToLog("Training for constraint #" + std::to_string(index) + " completed successfully.");
        } catch (const std::exception& e) {
            printInfoToLog("Error training surrogate model for constraint #" + std::to_string(index) + ": " + e.what());
        }

        ++index;
    }

    printInfoToLog("Model training for all constraints is done.");
}

void Optimizer::computeConstraintsandPenaltyTerm(Design &d) {

	if(isConstrained()){
		printInfoToLog("Evaluating constraints...");

		evaluateConstraints(d);

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);
		if(!ifConstraintsSatisfied){
			printInfoToLog("The new sample does not satisfy all the constraints");
			d.isDesignFeasible = false;

		}
		else{
			printInfoToLog("The new sample satisfies all the constraints");
			d.isDesignFeasible = true;
		}
		printInfoToLog("Evaluation of the constraints is ready...");
	}

}

void Optimizer::addConstraintValuesToData(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		if(!it->isUserDefinedFunction()){
			it->addDesignToData(d);
		}
	}

}

void Optimizer::printConstraints(void) const{

	std::cout<< "List of constraints = \n";

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){
		it->print();
	}

}


} /* Namespace Rodop */

