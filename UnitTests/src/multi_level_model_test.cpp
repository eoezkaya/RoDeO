/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-202 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */

#include "multi_level_method.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>



class MultiLevelModelTest : public ::testing::Test {
protected:
	void SetUp() override {

		system("mkdir MultiLevelModelTest");
		chdir("./MultiLevelModelTest");

		testModel.setName("MLtestModel");

		generateHimmelblauDataMultiFidelity("highFidelityTrainingData.csv","lowFidelityTrainingData.csv",nSamplesHiFi,nSamplesLowFi);
		testModel.setinputFileNameHighFidelityData("highFidelityTrainingData.csv");
		testModel.setinputFileNameLowFidelityData("lowFidelityTrainingData.csv");

		testModelWithShuffledData.setName("MLtestModel");
		generateHimmelblauDataMultiFidelityWithShuffle("highFidelityTrainingDataShuffled.csv","lowFidelityTrainingDataShuffled.csv",nSamplesHiFi,nSamplesLowFi);
		testModelWithShuffledData.setinputFileNameHighFidelityData("highFidelityTrainingDataShuffled.csv");
		testModelWithShuffledData.setinputFileNameLowFidelityData("lowFidelityTrainingDataShuffled.csv");

		lowerBounds = zeros<vec>(dim);
		upperBounds = zeros<vec>(dim);
		lowerBounds.fill(-6.0);
		upperBounds.fill(6.0);


		testModel.setBoxConstraints(lowerBounds,upperBounds);
		testModelWithShuffledData.setBoxConstraints(lowerBounds,upperBounds);



	}

	void TearDown() override {

		chdir("../");
		system("rm MultiLevelModelTest -r");


	}


	MultiLevelModel testModel;
	MultiLevelModel testModelWithShuffledData;
	unsigned int dim = 2;
	vec lowerBounds;
	vec upperBounds;
	unsigned int nSamplesLowFi = 200;
	unsigned int nSamplesHiFi  = 50;

};


TEST_F(MultiLevelModelTest, testMLModelInitialSettings) {

	ASSERT_FALSE(testModel.ifDataIsRead);
	ASSERT_FALSE(testModel.ifErrorDataIsSet);
	ASSERT_FALSE(testModelWithShuffledData.ifDataIsRead);
	ASSERT_FALSE(testModelWithShuffledData.ifErrorDataIsSet);
	ASSERT_FALSE(testModel.ifInitialized);
	ASSERT_FALSE(testModelWithShuffledData.ifInitialized);


}



TEST_F(MultiLevelModelTest, testAddNewSampleToData){

	testModel.initializeSurrogateModel();
	//	testModel.setNumberOfTrainingIterations(1000);
	//	testModel.setNumberOfMaximumIterationsForGammaTraining(10);
	//	testModel.train();

	rowvec newLowFiSample(dim+1);
	vec x(dim);
	x(0) = generateRandomDouble(lowerBounds(0),upperBounds(0) );
	x(1) = generateRandomDouble(lowerBounds(1),upperBounds(1) );

	double f = HimmelblauLowFi(x);
	newLowFiSample(0) = x(0);
	newLowFiSample(1) = x(1);
	newLowFiSample(2) = f;

	testModel.addNewSampleToData(newLowFiSample);

	ASSERT_EQ(testModel.getNumberOfLowFiSamples(),201);


}

TEST_F(MultiLevelModelTest, testAddNewHiFiSampleToData){

	testModel.initializeSurrogateModel();
	//	testModel.setNumberOfTrainingIterations(1000);
	//	testModel.setNumberOfMaximumIterationsForGammaTraining(10);
	//	testModel.train();

	rowvec newHiFiSample(dim+1);
	rowvec newLowFiSample(dim+1);
	vec x(dim);
	x(0) = generateRandomDouble(lowerBounds(0),upperBounds(0) );
	x(1) = generateRandomDouble(lowerBounds(1),upperBounds(1) );

	double f = HimmelblauLowFi(x);
	newLowFiSample(0) = x(0);
	newLowFiSample(1) = x(1);
	newLowFiSample(2) = f;

	f = Himmelblau(x);
	newHiFiSample(0) = x(0);
	newHiFiSample(1) = x(1);
	newHiFiSample(2) = f;


	testModel.addNewSampleToData(newLowFiSample);
	testModel.addNewHiFiSampleToData(newHiFiSample);

	ASSERT_EQ(testModel.getNumberOfHiFiSamples(),51);


}




TEST_F(MultiLevelModelTest, testReadData){

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();
	testModel.readData();
	ASSERT_TRUE(testModel.ifDataIsRead);

}


TEST_F(MultiLevelModelTest, testfindIndexHiFiToLowFiData){



	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	unsigned int index = testModel.findIndexHiFiToLowFiData(2);


	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();


	rowvec dx = dataLowFi.row(index) - dataHiFi.row(2);

	for(unsigned int i=0; i<dim; i++){

		EXPECT_LT(dx(i),10E-10);
	}

}


TEST_F(MultiLevelModelTest, testPrepareErrorData){

	testModel.bindLowFidelityModel();
	testModel.bindErrorModel();


	testModel.readHighFidelityData();
	testModel.readLowFidelityData();
	testModel.setDimensionsHiFiandLowFiModels();

	testModel.prepareErrorData();

	mat dataLowFi = testModel.getRawDataLowFidelity();
	mat dataHiFi = testModel.getRawDataHighFidelity();
	mat dataError = testModel.getRawDataError();

	double difference = fabs( dataError(4,2) - (dataHiFi(4,2) - dataLowFi(4,2) ));

	EXPECT_LT(difference, 10E-08);
	ASSERT_TRUE(testModel.ifErrorDataIsSet);


}

TEST_F(MultiLevelModelTest, testPrepareErrorDataWithShuffle){


	testModelWithShuffledData.bindLowFidelityModel();
	testModelWithShuffledData.bindErrorModel();


	testModelWithShuffledData.readHighFidelityData();
	testModelWithShuffledData.readLowFidelityData();
	testModelWithShuffledData.setDimensionsHiFiandLowFiModels();

	testModelWithShuffledData.prepareErrorData();

	mat dataLowFi = testModelWithShuffledData.getRawDataLowFidelity();
	mat dataHiFi = testModelWithShuffledData.getRawDataHighFidelity();
	mat dataError = testModelWithShuffledData.getRawDataError();

	unsigned int someIndexInData = 2;
	unsigned int index = testModelWithShuffledData.findIndexHiFiToLowFiData(someIndexInData);


	double difference = fabs( dataError(someIndexInData,dim) - (dataHiFi(someIndexInData,dim) -dataLowFi(index,dim) ));

	EXPECT_LT(difference, 10E-08);
	ASSERT_TRUE(testModelWithShuffledData.ifErrorDataIsSet);



}


TEST_F(MultiLevelModelTest, testfindNearestNeighbourLowFidelity){

	testModel.initializeSurrogateModel();


	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTrainingData.csv",csv_ascii);

	unsigned int someIndex = 14;
	rowvec x = samplesLowFi.row(someIndex);
	rowvec xp(dim); xp(0) = x(0) + 0.001;  xp(1) = x(1)-0.001;

	xp = normalizeRowVector(xp, lowerBounds, upperBounds);

	unsigned int indx = testModel.findNearestNeighbourLowFidelity(xp);

	EXPECT_EQ(indx, someIndex);


}


TEST_F(MultiLevelModelTest, testfindNearestL1DistanceToALowFidelitySample){

	testModel.initializeSurrogateModel();
	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTrainingData.csv",csv_ascii);

	rowvec x = samplesLowFi.row(14);
	rowvec xp(dim); xp(0) = x(0);  xp(1) = x(1);

	xp = normalizeRowVector(xp, lowerBounds, upperBounds);
	xp(0) += 0.0001;
	xp(1) -= 0.0001;

	double dist = testModel.findNearestL1DistanceToALowFidelitySample(xp);
	EXPECT_LT(fabs(dist-0.0002),10E-08 );

}



TEST_F(MultiLevelModelTest, testfindNearestNeighbourHighFidelity){

	testModel.initializeSurrogateModel();


	mat samples;
	samples.load("highFidelityTrainingData.csv",csv_ascii);

	unsigned int someIndex = 14;
	rowvec x = samples.row(someIndex);
	rowvec xp(2); xp(0) = x(0) + 0.001;  xp(1) = x(1)-0.001;

	xp = normalizeRowVector(xp, lowerBounds, upperBounds);

	unsigned int indx = testModel.findNearestNeighbourHighFidelity(xp);

	EXPECT_EQ(indx, someIndex);


}

TEST_F(MultiLevelModelTest, testtrainLowFidelityModel){

	testModel.initializeSurrogateModel();
	testModel.setNumberOfTrainingIterations(1000);
	testModel.trainLowFidelityModel();

	mat samplesLowFi;
	samplesLowFi.load("lowFidelityTrainingData.csv",csv_ascii);


	double SE = 0.0;
	for(int i=0; i<nSamplesLowFi; i++){

		rowvec x = samplesLowFi.row(i);
		rowvec xIn(dim);
		xIn(0) = x(0) + 0.01;
		xIn(1) = x(1) + 0.01;

		double f = HimmelblauLowFi(xIn.memptr());
		xIn = normalizeRowVector(xIn, lowerBounds, upperBounds);

		double ftilde = testModel.interpolateLowFi(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/nSamplesLowFi;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 200.0);




}

TEST_F(MultiLevelModelTest, testtrainLowFidelityModelWithGradient){



	generateHimmelblauDataMultiFidelityWithGradients("highFidelityTrainingDataWithGradient.csv","lowFidelityTrainingDataWithGradient.csv",nSamplesHiFi ,nSamplesLowFi);


	testModel.setinputFileNameHighFidelityData("highFidelityTrainingDataWithGradient.csv");
	testModel.setinputFileNameLowFidelityData("lowFidelityTrainingDataWithGradient.csv");


	testModel.setGradientsOn();
	testModel.setGradientsOnLowFi();
	testModel.setGradientsOnHiFi();


	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.setNumberOfMaximumIterationsForGammaTraining(10);
	testModel.trainLowFidelityModel();


	double SE = 0.0;
	for(int i=0; i<nSamplesLowFi; i++){

		rowvec x(dim);

		x = generateRandomRowVector(lowerBounds,upperBounds);

		rowvec xIn(2);
		xIn(0) = x(0);
		xIn(1) = x(1);

		double f = HimmelblauLowFi(xIn.memptr());
		xIn = normalizeRowVector(xIn, lowerBounds, upperBounds);

		double ftilde = testModel.interpolateLowFi(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/100;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);


}

TEST_F(MultiLevelModelTest, testTrainErrorModel){

	testModel.initializeSurrogateModel();


	testModel.setNumberOfTrainingIterations(1000);
	testModel.setNumberOfMaximumIterationsForGammaTraining(10);
	testModel.trainErrorModel();

	mat samplesError;
	samplesError.load("MLtestModel_Error.csv",csv_ascii);


	double squaredError = 0.0;
	for(int i=0; i<nSamplesHiFi; i++){

		rowvec x = samplesError.row(i);
		double xp[2]; xp[0] = x(0)+0.1; xp[1] = x(1)+0.1;
		rowvec xIn(2);
		xIn(0) = x(0)+0.1;
		xIn(1) = x(1)+0.1;

		xIn = normalizeRowVector(xIn, lowerBounds, upperBounds);
		double ftilde = testModel.interpolateError(xIn);
		double f = -Waves2D(xp)*50.0;
		squaredError+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"ftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n\n";
#endif


	}


	double MSE = squaredError/nSamplesHiFi;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);


}


TEST_F(MultiLevelModelTest, testDetermineGammaBasedOnData){

	testModel.initializeSurrogateModel();
	testModel.setNumberOfMaximumIterationsForGammaTraining(100);
	testModel.determineGammaBasedOnData();
	double gamma = testModel.getGamma();

	EXPECT_GT(gamma,-0.1);
	EXPECT_LT(gamma,10.1);

}



TEST_F(MultiLevelModelTest, testInterpolate){

	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(1000);


	testModel.train();

	double SE = 0.0;
	for(int i=0; i<100; i++){

		rowvec x(2);
		x = generateRandomRowVector(lowerBounds, upperBounds);
		rowvec xIn(2);
		xIn(0) = x(0);
		xIn(1) = x(1);

		double f = Himmelblau(xIn.memptr());
		xIn = normalizeRowVector(xIn, lowerBounds, upperBounds);

		double ftilde = testModel.interpolate(xIn);

		SE+= (f-ftilde)*(f-ftilde);

#if 0
		std::cout<<"\nftilde = "<<ftilde<<"\n";
		std::cout<<"f      = "<<f<<"\n";
		std::cout<<"SE = "<<(f-ftilde)*(f-ftilde)<<"\n\n";
#endif

	}

	double MSE = SE/100;

#if 0
	std::cout<<"MSE = "<<MSE<<"\n";
#endif
	EXPECT_LT(MSE, 1000.0);



}



TEST_F(MultiLevelModelTest, testInterpolateWithVariance){

	testModel.initializeSurrogateModel();

	testModel.setNumberOfTrainingIterations(500);

	testModel.train();

	mat XLowFi = testModel.getRawDataLowFidelity();
	mat XHiFi = testModel.getRawDataHighFidelity();

	unsigned int someIndex = 3;
	unsigned int index = testModel.findIndexHiFiToLowFiData(someIndex);

	rowvec sampleLF = XLowFi.row(index);
	rowvec sampleHF = XHiFi.row(someIndex);

	sampleLF.print();
	sampleHF.print();

	double f = sampleHF(dim);

	rowvec xp(dim); xp(0) = sampleHF(0); xp(1) = sampleHF(1);

	xp = normalizeRowVector(xp, lowerBounds, upperBounds);

	double ftilde = 0.0;
	double ssqr   = 10.0;

	testModel.interpolateWithVariance(xp, &ftilde, &ssqr);

	printScalar(ftilde);
	printScalar(f);
	printScalar(ssqr);

	double errorInf = fabs(f-ftilde);

	EXPECT_LT(errorInf,10E-06);
	EXPECT_LT(ssqr,10E-06);


}

TEST_F(MultiLevelModelTest, testCalculateExpectedImprovement){


	testModel.initializeSurrogateModel();
	testModel.setNumberOfTrainingIterations(1000);
	testModel.train();


	mat XLowFi = testModel.getRawDataLowFidelity();
	mat XHiFi = testModel.getRawDataHighFidelity();

	double fmin = LARGE;
	unsigned int minIndex = -1;
	rowvec minSample;
	for(unsigned int i=0; i<XHiFi.n_rows; i++){

		rowvec sampleHF = XHiFi.row(i);


		if(sampleHF(dim) < fmin){

			fmin = sampleHF(dim);
			minIndex = i;
			minSample = sampleHF;
		}

	}

	rowvec xp(dim); xp(0) = minSample(0); xp(1) = minSample(1);
	xp = normalizeRowVector(xp, lowerBounds, upperBounds);

	xp += 0.01;

	CDesignExpectedImprovement testDesign(xp);

	testModel.calculateExpectedImprovement(testDesign);

	EXPECT_GT(testDesign.valueExpectedImprovement, 10E-5);

}


TEST(MultiLevelModelTestNACA0012, testMLModel){

	chdir("./MultiLevelModelTestNACA0012");

	MultiLevelModel testModel;
	testModel.setName("MLtestModel");

	testModel.setinputFileNameHighFidelityData("CD_HiFi.csv");
	testModel.setinputFileNameLowFidelityData("CD_LowFi.csv");

	unsigned int dim = 38;

	vec lowerBounds = zeros<vec>(dim);
	vec upperBounds = zeros<vec>(dim);
	lowerBounds.fill(-0.00001);
	upperBounds.fill(0.00001);

	mat CLValidation;

	CLValidation.load("CD.csv", csv_ascii);

	mat X = CLValidation.submat(0,0,99,dim-1);
	vec CL = CLValidation.col(dim);

	CL.print();

	mat Xnormalized = normalizeMatrix(X,lowerBounds,upperBounds);
	Xnormalized = Xnormalized*(1.0/dim);
	Xnormalized.print();


	mat results(100, 3);

	testModel.setDisplayOn();
	testModel.setBoxConstraints(lowerBounds,upperBounds);
	testModel.initializeSurrogateModel();
	testModel.setNumberOfThreads(4);
	testModel.train();

	double SE = 0.0;
	for(unsigned int i=0; i<100; i++){

		rowvec xp= Xnormalized.row(i);

		double ftilde = testModel.interpolate(xp);
		double f = CL(i);
		results(i,0)  = ftilde;
		results(i,1)  = f;
		results(i,2)  = (f - ftilde) * (f - ftilde);
		SE += (f - ftilde) * (f - ftilde);

	}

	SE = SE/100;

	results.save("results.csv", csv_ascii);
	printScalar(SE);



}



