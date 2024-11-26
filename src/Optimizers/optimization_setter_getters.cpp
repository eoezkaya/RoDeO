#include "./INCLUDE/optimization.hpp"
#ifdef OPENMP_SUPPORT
#include <omp.h>
#endif


#include <cassert>

namespace Rodop{

void Optimizer::setName(std::string problemName){
	name = problemName;
}

void Optimizer::setMaximumNumberOfIterations(unsigned int maxIterations){

	maxNumberOfSamples = maxIterations;

}


void Optimizer::setMaximumNumberOfIterationsLowFidelity(unsigned int maxIterations){

	maxNumberOfSamplesLowFidelity =  maxIterations;


}



void Optimizer::setMaximumNumberOfInnerIterations(unsigned int maxIterations){

	iterMaxAcquisitionFunction = maxIterations;
	printInfoToLog("NUmber of inner iterations is set to = " + std::to_string(iterMaxAcquisitionFunction));

}


void Optimizer::setFileNameDesignVector(std::string filename){

	if (filename.empty() || filename.find_first_not_of(' ') == std::string::npos) {
		throw std::invalid_argument("Filename cannot be empty or just whitespace.");
	}
	designVectorFileName = filename;

}


void Optimizer::setBoxConstraints(const Bounds& boxConstraints) {
	lowerBounds = boxConstraints.getLowerBounds();
	upperBounds = boxConstraints.getUpperBounds();

	if (lowerBounds.getSize() != upperBounds.getSize()) {
		throw std::invalid_argument("Lower and upper bounds must have the same size.");
	}

	for (unsigned int i = 0; i < lowerBounds.getSize() ; ++i) {
		if (lowerBounds(i) > upperBounds(i)) {
			throw std::invalid_argument("Each lower bound must be less than or equal to the corresponding upper bound.");
		}
	}

	assert(ifObjectFunctionIsSpecied && "Objective function must be specified before setting box constraints.");
	objFun.setParameterBounds(boxConstraints);

	if (isConstrained()) {
		for (auto& constraintFunction : constraintFunctions) {
			constraintFunction.setParameterBounds(boxConstraints);
		}
	}

	globalOptimalDesign.setBoxConstraints(boxConstraints);
	ifBoxConstraintsSet = true;
}


void Optimizer::setUseTangentsOn(void){
	ifTangentsAreUsed = true;
}

#ifdef OPENMP_SUPPORT
void Optimizer::setNumberOfThreads(unsigned int n) {

	if (n == 0) {
		printErrorToLog("Invalid number of threads: 0");
		throw std::invalid_argument("Number of threads must be greater than 0.");
	}

	numberOfThreads = n;
	printInfoToLog("Number of threads set to: " + std::to_string(n));

}

#endif

void Optimizer::setDimension(unsigned int dim){

	dimension = dim;
	lowerBounds.resize(dimension);
	upperBounds.resize(dimension);
	initializeBoundsForAcquisitionFunctionMaximization();
	iterMaxAcquisitionFunction = dimension*10000;

	globalOptimalDesign.setDimension(dim);
	currentBestDesign.setDimension(dim);
	history.setDimension(dim);

}


void Optimizer::setHowOftenTrainModels(unsigned int value){
	howOftenTrainModels = value;
}

void Optimizer::setImprovementPercentThresholdForGradientStep(double value){
	assert(value>=0);
	assert(value<=100);
	improvementPercentThresholdForGradientStep = value;
}

void Optimizer::setAPIUseOn(void){
	ifAPIisUsed = true;
}

void Optimizer::setParameterNames(std::vector<std::string> names) {
	history.setParameterNames(names);
}

} /* Namespace Rodop */
