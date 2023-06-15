#include "kriging_training.hpp"
#include "LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "test_functions.hpp"
#include "test_defines.hpp"
#include<gtest/gtest.h>

#ifdef TEST_KRIGINGGENERALIZATION


std::vector<double> meanErrors;
std::vector<std::string> problemNames;

class KrigingModelGeneralizationTests : public ::testing::Test {
protected:
	void SetUp() override {





	}

	void TearDown() override {



	}






	KrigingModel testModel;
	mat trainingData;
	mat testData;
};


TEST_F(KrigingModelGeneralizationTests, testRosenbrock) {

	unsigned int NTraining = 100;
	unsigned int NTest     = 10000;
	unsigned int numberOfExperiments = 20;
	TestFunction testFunction("Rosenbrock",2);
	testFunction.setBoxConstraints(-5.0,10.0);
	testFunction.func_ptr = Rosenbrock;
	testFunction.filenameTrainingData = "Rosenbrock.csv";
	testFunction.filenameTestData = "RosenbrockTest.csv";
	testFunction.numberOfTrainingSamples = NTraining;
	testFunction.numberOfTestSamples = NTest;

	testFunction.generateTestSamples();
	testData = testFunction.testSamples;

	double sumError = 0.0;
	for(unsigned int i=0; i<numberOfExperiments; i++){


		testModel.setName("Rosenbrock");
		testModel.setNameOfInputFile("Rosenbrock.csv");
		testModel.readData();
		testModel.setBoxConstraints(0.0, 200.0);
		testModel.normalizeData();
		testModel.initializeSurrogateModel();

		testModel.setNumberOfThreads(2);
		testModel.setNumberOfTrainingIterations(1000);
		testModel.train();


		testModel.setNameOfInputFileTest("RosenbrockTest.csv");
		testModel.readDataTest();
		testModel.normalizeDataTest();


		double outSampleError = testModel.calculateOutSampleError();
		printScalar(outSampleError);
		sumError+= outSampleError;

	}

	double meanError = sumError/numberOfExperiments;
	meanErrors.push_back(meanError);
	problemNames.push_back("Rosenbrock");

}


//TEST_F(KrigingModelGeneralizationTests, testEggholder) {
//
//	unsigned int NTraining = 100;
//	unsigned int NTest     = 10000;
//	unsigned int numberOfExperiments = 20;
//	TestFunction testFunction("Eggholder",2);
//
//
//
//	testFunction.setFunctionPointer(Eggholder);
//
//	double sumError = 0.0;
//	for(unsigned int i=0; i<numberOfExperiments; i++){
//
//		testFunction.setBoxConstraints(0,200.0);
//		trainingData = testFunction.generateRandomSamples(NTraining);
//		saveMatToCVSFile(trainingData,"Eggholder.csv");
//		testData = testFunction.generateRandomSamples(NTest);
//		saveMatToCVSFile(testData,"EggholderTest.csv");
//
//
//		testModel.setName("Eggholder");
//		testModel.setNameOfInputFile("Eggholder.csv");
//		testModel.readData();
//		testModel.setBoxConstraints(0.0, 200.0);
//		testModel.normalizeData();
//		testModel.initializeSurrogateModel();
//
//		testModel.setNumberOfThreads(2);
//		testModel.setNumberOfTrainingIterations(1000);
//		testModel.train();
//
//		testModel.setNameOfInputFileTest("EggholderTest.csv");
//		testModel.readDataTest();
//		testModel.normalizeDataTest();
//
//
//		double outSampleError = testModel.calculateOutSampleError();
//		sumError+= outSampleError;
//
//	}
//
//	double meanError = sumError/numberOfExperiments;
//	meanErrors.push_back(meanError);
//	problemNames.push_back("Eggholder");
//
//}




TEST_F(KrigingModelGeneralizationTests, allKrigingGeneralizationTests) {

	int numberOfProblemsTried = meanErrors.size();

	std::cout<<"Number of problems tried = "<<numberOfProblemsTried<<"\n";

	for(int i=0; i<numberOfProblemsTried;i++){

		std::cout<<problemNames.at(i)<< "     " << meanErrors.at(i) <<"\n";

	}



}

#endif

