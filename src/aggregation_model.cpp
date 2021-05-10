/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include <armadillo>
#include <random>
#include <map>
using namespace arma;


#include "aggregation_model.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "linear_regression.hpp"

AggregationModel::AggregationModel():SurrogateModel(){}


AggregationModel::AggregationModel(std::string name):SurrogateModel(name),krigingModel(name) {

	modelID = AGGREGATION;
	hyperparameters_filename = label + "_aggregation_model_hyperparameters.csv";
	this->ifHasGradientData = true;


}

void AggregationModel::determineRhoBasedOnData(void){


	unsigned int numberOfProbles = 1000;
	vec probe_distances_sample(numberOfProbles);

	for(unsigned int i=0; i<numberOfProbles; i++){

		rowvec xp = generateRandomRowVector(0.0, 1.0/dim, dim);

		int indxNearestNeighbor= findNearestNeighbor(xp);
		double distance = calculateL1norm(xp - X.row(indxNearestNeighbor));

#if 0
		printf("point:\n");
		xp.print();
		printf("nearest neighbour is:\n");
		X.row(indxNearestNeighbor).print();
		printf("minimum distance (L1 norm)= %10.7f\n\n",distance);

#endif

		probe_distances_sample(i)= distance;

	}

	double average_distance_sample = mean(probe_distances_sample);
#if 1
	printf("Maximum distance (L1 norm) = %10.7f\n", max(probe_distances_sample));
	printf("Minimum distance (L1 norm) = %10.7f\n", min(probe_distances_sample));
	printf("Average distance (L1 norm)= %10.7f\n", average_distance_sample);
#endif


	/* obtain gradient statistics */
	double sumnormgrad = 0.0;
	for(unsigned int i=0; i<N; i++){

		rowvec gradVec= gradientData.row(i);


		double normgrad= calculateL1norm(gradVec);
		sumnormgrad+= normgrad;

	}

	double avg_norm_grad= sumnormgrad/N;
#if 1
	printf("average norm grad = %10.7e\n", avg_norm_grad);
#endif
	/* define the search range for the hyperparameter r */



	/* Note that
	 *
	 *  w2 = exp(-rho*norm(distance)*norm(grad))
	 *
	 *
	 */


	rho = -2.0*log(0.0001)/ (max(probe_distances_sample) * avg_norm_grad);
//	rho = -log(0.05)/ (min(probe_distances_sample) * avg_norm_grad);

#if 1
	printf("rho = %10.7f\n",rho);
	printf("dual model weight at the maximum distance = %15.10f\n",exp(-rho*max(probe_distances_sample)* avg_norm_grad));
	printf("dual model weight at the average distance = %15.10f\n",exp(-rho*mean(probe_distances_sample)* avg_norm_grad));
	printf("dual model weight at the minimum distance = %15.10f\n",exp(-rho*min(probe_distances_sample)* avg_norm_grad));
#endif






}

void AggregationModel::initializeSurrogateModel(void){

	numberOfTrainingIterations = 10000;

	krigingModel.ifHasGradientData = true;

	krigingModel.readData();
	krigingModel.setParameterBounds(xmin,xmax);
	krigingModel.normalizeData();
	krigingModel.initializeSurrogateModel();

	readData();
	normalizeData();

	numberOfHyperParameters = dim + 1;

	L1NormWeights = zeros<vec>(dim);
	L1NormWeights.fill(1.0);



	ifInitialized = true;

#if 0
	printSurrogateModel();
#endif



}

void AggregationModel::prepareTrainingAndTestData(void){

	int NTestSet = 2*(N/5);


	int NTrainingSet = N - NTestSet;
#if 0
	cout << "NTestSet  = "<<NTestSet<<"\n";
	cout << "NTrainingSet = "<<NTrainingSet<<"\n";
#endif
	mat shuffledrawData = shuffle(rawData);

	mat trainingRawData = shuffledrawData.submat( 0, 0, NTrainingSet-1, 2*dim );
	mat testRawData = shuffledrawData.submat( NTrainingSet, 0, N-1, 2*dim );



	trainingDataForHyperParameterOptimization.ifHasGradientData = true;
	trainingDataForHyperParameterOptimization.fillWithData(trainingRawData);
	trainingDataForHyperParameterOptimization.normalizeAndScaleData(xmin,xmax);

	testDataForHyperParameterOptimization.ifHasGradientData = true;
	testDataForHyperParameterOptimization.fillWithData(testRawData);
	testDataForHyperParameterOptimization.normalizeAndScaleData(xmin,xmax);


}

PartitionData  AggregationModel::getTrainingData(void) const{

	return trainingDataForHyperParameterOptimization;
}

PartitionData AggregationModel::getTestData(void) const{

	return testDataForHyperParameterOptimization;
}



void AggregationModel::printSurrogateModel(void) const{

	krigingModel.printSurrogateModel();
	printVector(L1NormWeights, "L1NormWeights");
	std::cout<<"rho = "<<rho<<"\n";

}

void AggregationModel::printHyperParameters(void) const{

	cout << "\nHyperparameters of the aggregation model...\n";
	krigingModel.printHyperParameters();
	cout << "rho = " << rho << "\n";


}
void AggregationModel::saveHyperParameters(void) const{

	assert(numberOfHyperParameters == dim+1);

	vec saveBuffer(numberOfHyperParameters);
	saveBuffer(0) = rho;

	for(unsigned int i=1; i<dim+1; i++){

		saveBuffer(i) = L1NormWeights(i-1);

	}


	saveBuffer.save(hyperparameters_filename,csv_ascii);



}

void AggregationModel::loadHyperParameters(void){

	assert(numberOfHyperParameters == dim+1);

	vec loadBuffer(numberOfHyperParameters);
	loadBuffer.load(hyperparameters_filename,csv_ascii);

	if(loadBuffer.size() == numberOfHyperParameters){

		rho = loadBuffer(0);
		for(unsigned int i=1; i<dim+1; i++){

			L1NormWeights(i-1) = loadBuffer(i);

		}

	}
	else{

		std::cout<<"WARNING: Number of entries in the hyperparameter file is inconsistent with the problem dimension!\n";

	}




}

void AggregationModel::updateAuxilliaryFields(void){

	krigingModel.updateAuxilliaryFields();


}


vec AggregationModel::getL1NormWeights(void) const{

	return this->L1NormWeights;


}


void AggregationModel::determineOptimalL1NormWeights(void){

	prepareTrainingAndTestData();

	mat rawDataSave = rawData;

	modifyRawDataAndAssociatedVariables(trainingDataForHyperParameterOptimization.rawData);



	/* we set rho to zero because we want to use only dual model in the training */
	rho = 0.0;


	double minimumValidationError = LARGE;
	vec weightsBest(dim);

	for(unsigned int trainingIter=0; trainingIter<numberOfTrainingIterations; trainingIter++){

		if(trainingIter>0){

			generateRandomHyperParams();

		}



		tryModelOnTestSet(testDataForHyperParameterOptimization);
		double validationError = testDataForHyperParameterOptimization.calculateMeanSquaredError();
#if 0
		printf("validationError = %10.7f, minimumValidationError = %10.7f\n",validationError,minimumValidationError);
#endif

		if(validationError + 10E-06 < minimumValidationError ){
#if 0
			cout<<"A better set of hyper-parameters is found\n";
			cout<<"Error = "<<validationError<<"\n";
			printVector(L1NormWeights);
#endif
			minimumValidationError = validationError;
			weightsBest = L1NormWeights;

		}

	}



#if 0
	std::cout<<"Optimal values found:\n";
	printVector(weightsBest,"L1NormWeights");
#endif


	L1NormWeights = weightsBest;

	modifyRawDataAndAssociatedVariables(rawDataSave);



}



void AggregationModel::train(void){

	if(!ifInitialized){

		this->initializeSurrogateModel();
	}


	loadHyperParameters();
#if 0
	std::cout<<"Initial weights for the L1norm:\n";
	printVector(L1NormWeights,"L1NormWeights");
#endif


	krigingModel.train();


	determineOptimalL1NormWeights();

	determineRhoBasedOnData();

	saveHyperParameters();


}


void AggregationModel::generateRandomHyperParams(void){


	double sumWeights=0.0;
	for(unsigned int i=0; i<dim; i++){

		L1NormWeights(i) = generateRandomDouble(0.0, 1.0);
		sumWeights+= L1NormWeights(i);

	}

	for(unsigned int i=0; i<dim; i++){

		L1NormWeights(i) = (dim*L1NormWeights(i))/sumWeights;

	}




}

void AggregationModel::setRho(double rho){
	this->rho = rho;


}

/* Be very careful using this method */

void AggregationModel::modifyRawDataAndAssociatedVariables(mat dataMatrix){

	assert(dataMatrix.n_cols == rawData.n_cols);
	this->rawData = dataMatrix;
	N = rawData.n_rows;
	Xraw = rawData.submat(0, 0, N - 1, dim - 1);
	y = rawData.col(dim);
	if(ifHasGradientData){

		gradientData = rawData.submat(0, dim+1, N - 1, 2*dim);

	}

	X = normalizeMatrix(Xraw, xmin, xmax);
	X = (1.0/dim)*X;
}

double AggregationModel::interpolate(rowvec x) const{

	/* find the closest seeding point to the xp in the data set */


	int indx = findNearestNeighbor(x);
	rowvec xNearestPoint = X.row(indx);
	rowvec xNearestPointRaw = Xraw.row(indx);
	rowvec gradNearestPoint = gradientData.row(indx);
	double yNearestPoint = y(indx);

	rowvec xp = normalizeRowVectorBack(x, xmin, xmax);



	rowvec xDiff = x - xNearestPoint;
	rowvec xDiffRaw = xp - xNearestPointRaw;

	double min_dist = calculateWeightedL1norm(xDiff, L1NormWeights);


	double fSurrogateKriging = krigingModel.interpolate(x);


	rowvec gradientVector = gradientData.row(indx);


	double normgrad = calculateL1norm(gradientVector);


	double fSurrogateDual = yNearestPoint+ dot(xDiffRaw,gradientVector);


	double w2 = exp(-rho*min_dist*normgrad);
	double w1 = 1.0 - w2;


	return w1*fSurrogateKriging+ w2* fSurrogateDual;


}



void AggregationModel::interpolateWithVariance(rowvec x,double *f_tilde,double *ssqr) const{



	/* find the closest seeding point to the xp in the data set */

	int indx = findNearestNeighbor(x);
	assert(indx < X.n_rows);

	if(indx > X.n_rows){

		std::cout<<"indx = "<<indx<<"\n";


	}

	rowvec xNearestPoint = X.row(indx);
	rowvec xNearestPointRaw = Xraw.row(indx);
	rowvec gradNearestPoint = gradientData.row(indx);
	double yNearestPoint = y(indx);

	rowvec xp = normalizeRowVectorBack(x, xmin, xmax);

#if 0
	printVector(x,"x");
	printVector(xp,"xp");
	cout <<"The closest point to x has an index = "<<indx<<":\n";
	printVector(xNearestPoint,"xNearestPoint");
	printVector(xNearestPointRaw,"xNearestPointRaw");
	std::cout<<"y = "<<yNearestPoint<<"\n";
#endif

	rowvec xDiff = x - xNearestPoint;
	rowvec xDiffRaw = xp - xNearestPointRaw;


#if 0
	printVector(xDiff,"xDiff");
	printVector(xDiffRaw,"xDiffRaw");
#endif
	double min_dist = calculateWeightedL1norm(xDiff, L1NormWeights);
#if 0
	cout<<"min_dist = "<<min_dist<<"\n";
#endif


	double fSurrogateKriging;

	krigingModel.interpolateWithVariance(x,&fSurrogateKriging,ssqr);
#if 0
	cout<<"fSurrogateKriging = "<<fSurrogateKriging<<"\n";
#endif


	rowvec gradientVector = gradientData.row(indx);
#if 0
	printVector(gradientVector,"gradientVector");
#endif

	double normgrad = calculateL1norm(gradientVector);
#if 0
	cout<<"normgrad = "<<normgrad<<"\n";
#endif

	double fSurrogateDual = yNearestPoint+ dot(xDiffRaw,gradientVector);
#if 0
	cout<<"fSurrogateDual = "<<fSurrogateDual<<"\n";

#endif

	double w2 = exp(-rho*min_dist*normgrad);
	double w1 = 1.0 - w2;
#if 0
	cout<<"w2 = "<<w2<<"\n";
	cout<<"w1 = "<<w1<<"\n";
	cout<<"result  = "<<w1*fSurrogateKriging+ w2* fSurrogateDual<<"\n";

#endif


	*f_tilde =  w1*fSurrogateKriging+ w2* fSurrogateDual;


}

void AggregationModel::calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const{


	double ftilde = 0.0;
	double ssqr   = 0.0;

	interpolateWithVariance(currentDesign.dv,&ftilde,&ssqr);

#if 0
	printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
#endif

	double	sigma = sqrt(ssqr)	;

#if 0
	printf("standart_ERROR = %15.10f\n",sigma);
#endif

	double expectedImprovementValue = 0.0;


	if(sigma!=0.0){


		double	Z = (ymin - ftilde)/sigma;
#if 0
		printf("EIfac = %15.10f\n",EIfac);
		printf("ymin = %15.10f\n",ymin);
#endif


		/* calculate the Expected Improvement value */
		expectedImprovementValue = (ymin - ftilde)*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
	}
	else{

		expectedImprovementValue = 0.0;

	}
#if 0
	printf("EI = %15.10f\n",EI);
#endif
	currentDesign.objectiveFunctionValue = ftilde;
	currentDesign.valueExpectedImprovement = expectedImprovementValue;

}



unsigned int AggregationModel::findNearestNeighbor(const rowvec &xp) const{

	assert(ifInitialized);
	unsigned int index = 0;
	double minL1Distance = LARGE;



	for(unsigned int i=0; i<X.n_rows; i++){

		rowvec x = X.row(i);

		rowvec xdiff = xp -x;

		double L1distance = calculateWeightedL1norm(xdiff, L1NormWeights);
		if(L1distance< minL1Distance){

			minL1Distance = L1distance;
			index = i;

		}

	}


	return index;


}



void AggregationModel::addNewSampleToData(rowvec newSample){


	assert(newSample.size() == 2*dim+1);

	/* avoid points that are too close to each other */

	bool flagTooClose=false;
	for(unsigned int i=0; i<N; i++){

		rowvec sample = rawData.row(i);

		if(checkifTooCLose(sample, newSample)) {

			flagTooClose = true;
		}

	}

	if(!flagTooClose){

		appendRowVectorToCSVData(newSample, filenameDataInput);

		updateModelWithNewData();

	}
	else{

		std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

	}

}

void AggregationModel::updateModelWithNewData(void){

	readData();
	normalizeData();

	krigingModel.updateModelWithNewData();

}



