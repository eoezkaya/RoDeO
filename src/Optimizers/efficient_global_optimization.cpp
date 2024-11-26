#include "./INCLUDE/optimization.hpp"
#include "./INCLUDE/aux.hpp"
#include<cassert>

namespace Rodop{


void Optimizer::performEfficientGlobalOptimizationOnlyWithFunctionalValues(void){

	while(1){

		statistics.getStartTimeIteration();

		outerIterationNumber++;
		printIterationNumber();


		if(outerIterationNumber%howOftenTrainModels == 1) {
			initializeSurrogates();
			trainSurrogates();
		}

		findTheMostPromisingDesignToBeSimulated();

		/* now make a simulation for the most promising design */

		printInfoToLog("Evaluating the objective function value for the new design...");
		objFun.evaluateDesign(currentBestDesign);
		statistics.numberOfObjectiveFunctionEvaluations++;

		if(isConstrained()){
			computeConstraintsandPenaltyTerm(currentBestDesign);
		}
		calculateImprovementValue(currentBestDesign);

		abortIfCurrentDesignHasANaN();

		printCurrentDesignToLogFile();

		printInfoToLog("Adding new sample to the objective function training data...");
		objFun.addDesignToData(currentBestDesign, "primal");

		if(isConstrained()){
			addConstraintValuesToData(currentBestDesign);
		}

		history.updateOptimizationHistory(currentBestDesign);

		if(currentBestDesign.improvementValue > globalOptimalDesign.improvementValue){

			string msg = "An improvement in objective function value has been achieved.";
			printInfoToLog(msg);

			trSuccessCounter++;
			trFailureCounter = 0;
			if(trSuccessCounter == trSuccessTolerance){
				trLength = trLength*2.0;
				if(trLength > trLengthMax){
					trLength = trLengthMax;
				}

				printInfoToLog("Enlarging the trust region. trLength = " + std::to_string(trLength));
				trSuccessCounter = 0;
			}

			double deltaImprovement = currentBestDesign.improvementValue - globalOptimalDesign.improvementValue;
			printInfoToLog("Delta improvement = ", deltaImprovement);

			if(deltaImprovement > bestDeltaImprovementValueAchieved){

				bestDeltaImprovementValueAchieved = deltaImprovement;
			}

			double percentImprovementRelativeToBest = (deltaImprovement/ bestDeltaImprovementValueAchieved)*100;
			//			printScalar(percentImprovementRelativeToBest);

			if(percentImprovementRelativeToBest > 10){
				printInfoToLog("Setting sigma factor again to 1.0.");
				sigmaMultiplier = 1.0;

			}

		}
		else{

			trFailureCounter++;
			trSuccessCounter = 0;

			if(trFailureCounter == trFailureTolerance){

				trLength = trLength/2.0;
				trFailureCounter = 0;
				if(trLength < trLengthMin){
					trLength = trLengthMin;
				}
				printInfoToLog("Shrinking the trust region. trLength = " + std::to_string(trLength));

			}


		}

		if(ifVariableSigmaStrategy){

			adjustSigmaFactor();
			sigmaMultiplier = sigmaMultiplier*sigmaGrowthFactor;
		}


		findTheGlobalOptimalDesign();

		printGlobalOptimalDesignToLogFile();

		statistics.getEndTimeIteration();
		statistics.evaluateElapsedSecondsForOptimizationIterationy();

		/* terminate optimization */
		if(outerIterationNumber >= maxNumberOfSamples){

			printInfoToLog("Number of function evaluations > maximum of function evaluations. Optimization is terminating...");
			printInfoToLog("Number of function evaluations = ", statistics.numberOfObjectiveFunctionEvaluations);
			break;
		}

	} /* end of the optimization loop */

}



void Optimizer::performEfficientGlobalOptimization(void){

	printInfoToLog("Starting global optimization procedure...");
	statistics.getStartTime();

	if(!ifObjectFunctionIsSpecied){
		printErrorToLog("Object function is not specified.");
		throw(std::runtime_error("Object function is not specified."));
	}

	if(!ifBoxConstraintsSet){
		printErrorToLog("Box constraint are not set.");
		throw(std::runtime_error("Box constraint are not set."));
	}

	checkIfSettingsAreOK();

	initializeOptimizerSettings();

	printSettingsToLogFile();

	initializeSurrogates();

	initializeCurrentBestDesign();
	initializeOptimizationHistory();


	performEfficientGlobalOptimizationOnlyWithFunctionalValues();


	statistics.getEndTime();
	statistics.evaluateElapsedSecondsForOptimizationStudy();
	printInfoToLog("Ending global optimization procedure...");
	printInfoToLog(statistics.generateElapsedTimeStringForOptimization());

}


} /* Namespace Rodop */
