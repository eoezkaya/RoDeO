#include "./INCLUDE/rodeo.h"
#include "./INCLUDE/rodeoapi.h"

namespace Rodop{

RobustDesignOptimizerAPI::RobustDesignOptimizerAPI() :impl(std::make_unique<RobustDesignOptimizer>()) {}
RobustDesignOptimizerAPI::~RobustDesignOptimizerAPI() = default;

void RobustDesignOptimizerAPI::print(void) {
	impl->print();
}

void RobustDesignOptimizerAPI::setName(const string nameInput) {
	impl->setName(nameInput);
}

void RobustDesignOptimizerAPI::setCurrentWorkingDirectory(string directory) {
	impl->setCurrentWorkingDirectory(directory);
}

void RobustDesignOptimizerAPI::setDoEStrategy(const std::string &input) {
	impl->setDoEStrategy(input);
}

void RobustDesignOptimizerAPI::setBoxConstraints(double *lb, double *ub) {
	impl->setBoxConstraints(lb, ub);
}

void RobustDesignOptimizerAPI::setNameOfTrainingDataFile(std::string name) {
	impl->setNameOfTrainingDataFile(name);
}

void RobustDesignOptimizerAPI::setDimension(unsigned int dim) {
	impl->setDimension(dim);
}

void RobustDesignOptimizerAPI::setObjectiveFunction(objectiveFunctionPtrType function, std::string name, std::string filename) {
	impl->setObjectiveFunction(function, name, filename);
}

void RobustDesignOptimizerAPI::addConstraint(objectiveFunctionPtrType function, std::string expression, std::string filename) {
	impl->addConstraint(function, expression, filename);
}

void RobustDesignOptimizerAPI::setDoEOn(unsigned int nSamples) {
	impl->setDoEOn(nSamples);
}

void RobustDesignOptimizerAPI::setMaxNumberOfFunctionEvaluations(unsigned int nSamples) {
	impl->setMaxNumberOfFunctionEvaluations(nSamples);
}

void RobustDesignOptimizerAPI::run(void) { impl->run(); }

}
