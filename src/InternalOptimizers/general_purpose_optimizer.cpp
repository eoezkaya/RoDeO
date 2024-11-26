#include <stdexcept> // For exceptions
#include <cmath>     // For std::sqrt and std::pow
#include "./INCLUDE/general_purpose_optimizer.hpp"


#ifdef OPENMP_SUPPORT
#include <omp.h>
#endif


namespace Rodop {

unsigned int GeneralPurposeOptimizer::getDimension(void) const {
    return dimension;
}

void GeneralPurposeOptimizer::setDimension(unsigned int dim) {
    dimension = dim;
}

void GeneralPurposeOptimizer::setBounds(double lb, double ub) {
    if (dimension == 0) {
        throw std::invalid_argument("Dimension must be specified first.");
    }

    lowerBounds.resize(dimension);
    lowerBounds.fill(lb);

    upperBounds.resize(dimension);
    upperBounds.fill(ub);
    ifBoundsAreSet = true;
}

void GeneralPurposeOptimizer::setBounds(std::vector<double> lb, std::vector<double> ub) {
    if (lb.size() != ub.size()) {
        throw std::invalid_argument("Lower bound and upper bound vectors must have the same size.");
    }
    if (lb.size() != dimension) {
        throw std::invalid_argument("Lower bound and upper bound vectors must match the dimension size.");
    }

    lowerBounds.fromStdVector(lb);
    upperBounds.fromStdVector(ub);

    ifBoundsAreSet = true;
}

void GeneralPurposeOptimizer::setBounds(const vec &lb, const vec &ub) {
    if (lb.getSize() != ub.getSize()) {
        throw std::invalid_argument("Lower bound and upper bound vectors must have the same size.");
    }
    if (lb.getSize() != dimension) {
        throw std::invalid_argument("Lower bound and upper bound vectors must match the dimension size.");
    }

    if(!(ub > lb)){
    	throw std::invalid_argument("Lower bound cannot be smaller than upper bounds.");
    }

    lowerBounds = lb;
    upperBounds = ub;

    ifBoundsAreSet = true;
}


void GeneralPurposeOptimizer::setDisplayOn(void) {
    ifScreenDisplay = true;
}

void GeneralPurposeOptimizer::setDisplayOff(void) {
    ifScreenDisplay = false;
}

void GeneralPurposeOptimizer::setMaxNumberOfFunctionEvaluations(unsigned int nMax) {
    if (nMax == 0) {
        throw std::invalid_argument("Maximum number of function evaluations must be greater than zero.");
    }
    maxNumberOfFunctionEvaluations = nMax;
}

void GeneralPurposeOptimizer::checkName(std::string name) const {
    if (name.empty()) {
        throw std::invalid_argument("The 'name' string must not be empty.");
    }
}

void GeneralPurposeOptimizer::setProblemName(std::string name) {
    checkName(name);
    problemName = name;
    ifProblemNameIsSet = true;
}

void GeneralPurposeOptimizer::setFilenameOptimizationHistory(std::string name) {
    checkName(name);
    filenameOptimizationHistory = name;
    ifFilenameOptimizationHistoryIsSet = true;
}

void GeneralPurposeOptimizer::setFilenameWarmStart(std::string name) {
    checkName(name);
    filenameWarmStart = name;
    ifFilenameWarmStartIsSet = true;
}

void GeneralPurposeOptimizer::setFilenameOptimizationResult(std::string name) {
    checkName(name);
    filenameOptimizationResult = name;
    ifFilenameOptimizationResultIsSet = true;
}

bool GeneralPurposeOptimizer::isProblemNameSet(void) {
    return ifProblemNameIsSet;
}

bool GeneralPurposeOptimizer::isFilenameOptimizationHistorySet(void) {
    return ifFilenameOptimizationHistoryIsSet;
}

bool GeneralPurposeOptimizer::isFilenameWarmStartSet(void) {
    return ifFilenameWarmStartIsSet;
}

bool GeneralPurposeOptimizer::isFilenameOptimizationResultSet(void) {
    return ifFilenameOptimizationResultIsSet;
}

void GeneralPurposeOptimizer::optimize(void) {
    throw std::logic_error("The 'optimize' method is not implemented.");
}

bool GeneralPurposeOptimizer::isObjectiveFunctionSet(void) const {
    return ifObjectiveFunctionIsSet;
}

bool GeneralPurposeOptimizer::areBoundsSet(void) const {
    return ifBoundsAreSet;
}

void GeneralPurposeOptimizer::setObjectiveFunction(GeneralPurposeObjectiveFunction functionToSet) {
    if (functionToSet == nullptr) {
        throw std::invalid_argument("Objective function cannot be null.");
    }
    calculateObjectiveFunction = functionToSet;
    ifObjectiveFunctionIsSet = true;
}



double GeneralPurposeOptimizer::callObjectiveFunction(const vec &x) {
    if (isObjectiveFunctionSet()) {
        return calculateObjectiveFunction(x.getPointer());
    } else {
        return calculateObjectiveFunctionInternal(x);
    }
}

double GeneralPurposeOptimizer::calculateObjectiveFunctionInternal(const vec &x) {
	x.print();
    throw std::logic_error("The internal objective function calculation is not implemented.");
}

void GeneralPurposeOptimizer::writeWarmRestartFile(void) {
    throw std::logic_error("The 'writeWarmRestartFile' method is not implemented.");
}

} // namespace Rodop
