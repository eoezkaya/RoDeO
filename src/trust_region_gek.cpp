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


#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxiliary_functions.hpp"

#ifdef GPU_VERSION
#include "kernel_regression_cuda.h"
#endif
#include "kernel_regression.hpp"

#include "Rodeo_globals.hpp"
#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"
#include "trust_region_gek.hpp"


AggregationModel::AggregationModel():SurrogateModel(){}


AggregationModel::AggregationModel(std::string name):SurrogateModel(name),kernelRegressionModel(name),krigingModel(name) {

	modelID = AGGREGATION;
	hyperparameters_filename = label + "_aggregation_model_hyperparameters.csv";
	ifUsesGradientData = true;

}

void AggregationModel::initializeSurrogateModel(void){

	if(label != "None"){

		numberOfIterForRhoOptimization = 1000;
		rho = 0.5;

		kernelRegressionModel.ifUsesGradientData = true;
		krigingModel.ifUsesGradientData = true;
		kernelRegressionModel.initializeSurrogateModel();
		krigingModel.initializeSurrogateModel();

		ReadDataAndNormalize();

		int NvalidationSet = N/5;
		int Ntraining = N - NvalidationSet;

		cout << "N = "<<N<<"\n";
		cout << "Ntraining = "<<Ntraining<<"\n";

		/* divide data into training and validation data, validation data is used for optimizing regularization parameters */

		mat shuffledrawData = rawData;
		if(NvalidationSet > 0){

			shuffledrawData = shuffle(rawData);
		}

		mat dataTraining  = shuffledrawData.submat( 0, 0, Ntraining-1, 2*dim );

		trainingData.ifHasGradientData = true;
		trainingData.fillWithData(dataTraining);
		trainingData.normalizeAndScaleData(xmin,xmax);

		if(NvalidationSet > 0){

			mat dataValidation    = shuffledrawData.submat( Ntraining, 0, N-1, 2*dim );
			testDataForRhoOptimizationLoop.ifHasGradientData = true;
			testDataForRhoOptimizationLoop.fillWithData(dataValidation);
			testDataForRhoOptimizationLoop.normalizeAndScaleData(xmin,xmax);


		}

	}


	unsigned int numberOfProbles = 1000;
	vec probe_distances_sample(numberOfProbles);

	for(unsigned int i=0; i<numberOfProbles; i++){

		rowvec xp = generateRandomRowVector(0.0, 1.0/dim, dim);

		int indxNearestNeighbor= findNearestNeighbor(xp);
		double distance = calculateL1norm(xp - X.row(indxNearestNeighbor));

#if 1
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
	printf("Average distance (L1 norm)= %10.7f\n\n", average_distance_sample);
#endif


	/* obtain gradient statistics */
	double sumnormgrad = 0.0;
	for(unsigned int i=0; i<N; i++){

		rowvec gradVec= this->gradientData.row(i);


		double normgrad= calculateL1norm(gradVec);
		sumnormgrad+= normgrad;

	}

	double avg_norm_grad= sumnormgrad/N;
#if 1
	printf("average norm grad = %10.7e\n", avg_norm_grad);
#endif
	/* define the search range for the hyperparameter r */


	double min_rho = 0.0;
	double max_rho = -2*log(0.00001)/ (max(probe_distances_sample) * avg_norm_grad);
	dr = (max_rho - min_rho)/(this->numberOfIterForRhoOptimization);

#if 1
	printf("average distance in the training data= %10.7f\n",average_distance_sample);
	printf("max_r = %10.7f\n",max_rho);
	printf("dr = %10.7f\n",dr);
	printf("factor at maximum distance at max r = %15.10f\n",exp(-max_rho*max(probe_distances_sample)* avg_norm_grad));
#endif


	ifInitialized = true;

#if 1
	printSurrogateModel();
#endif



}

void AggregationModel::printSurrogateModel(void) const{

	kernelRegressionModel.printSurrogateModel();
	krigingModel.printSurrogateModel();


}

void AggregationModel::printHyperParameters(void) const{

	cout << "\nHyperparameters of the aggregation model...\n";
	kernelRegressionModel.printHyperParameters();
	krigingModel.printHyperParameters();
	cout << "rho = " << rho << "\n";


}
void AggregationModel::saveHyperParameters(void) const{

	cout<<"Saving hyperparameters to the file:"<<hyperparameters_filename<<"\n";
	vec saveBuffer(dim*dim+1+2*dim+1+dim+1, fill::zeros);

	unsigned int count = 0;

	saveBuffer(count) = rho;
	cout<<"rho = "<<saveBuffer(count)<<"\n";
	count++;

	mat M = kernelRegressionModel.getMahalanobisMatrix();


	printMatrix(M,"M");
	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++) {


			saveBuffer(count) = M(i,j);
			count++;
		}


	saveBuffer(count) = kernelRegressionModel.getsigmaGaussianKernel();
	cout<<"sigma = "<<saveBuffer(count)<<"\n";
	count++;
	vec krigingWeights = krigingModel.getKrigingWeights();
	printVector(krigingWeights,"krigingWeights");
	vec regressionWeights = krigingModel.getRegressionWeights();
	printVector(regressionWeights,"regressionWeights");
	for(unsigned int i=0; i<2*dim; i++){
		saveBuffer(count) = krigingWeights(i);
		count++;
	}
	for(unsigned int i=0; i<dim+1; i++){
		saveBuffer(count) = regressionWeights(i);
		count++;
	}


	saveBuffer.save(hyperparameters_filename, csv_ascii);



}

void AggregationModel::loadHyperParameters(void){

	cout<<"Loading hyperparameters from the file:"<<hyperparameters_filename<<"\n";

	vec loadBuffer(dim*dim+1+2*dim+1+dim+1, fill::zeros);


	loadBuffer.load(hyperparameters_filename, csv_ascii);

	unsigned int count = 0;

	rho = loadBuffer(count);
	count++;

	cout<<"rho = "<<rho<<"\n";

	mat M(dim,dim);


	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++) {


			M(i,j) = loadBuffer(count);
			count++;
		}
	printMatrix(M,"M");

	double sigma = loadBuffer(count);
	count++;

	cout<<"sigma = "<<sigma<<"\n";

	kernelRegressionModel.setsigmaGaussianKernel(sigma);


	vec krigingWeights(2*dim);
	vec regressionWeights(dim+1);

	for(unsigned int i=0; i<2*dim; i++){
		krigingWeights(i) = loadBuffer(count);
		count++;
	}
	for(unsigned int i=0; i<dim+1; i++){
		regressionWeights(i) = loadBuffer(count);
		count++;
	}


	printVector(krigingWeights,"krigingWeights");
	printVector(regressionWeights,"regressionWeights");

	krigingModel.setKrigingWeights(krigingWeights);
	krigingModel.setRegressionWeights(regressionWeights);





}

void AggregationModel::updateAuxilliaryFields(void){

	krigingModel.updateAuxilliaryFields();


}

void AggregationModel::train(void){

	if(!ifInitialized){

		this->initializeSurrogateModel();
	}

	cout<<"Training Kriging model...\n";
	krigingModel.train();
	cout<<"Training Kernel regression model...\n";
	kernelRegressionModel.train();

    mat rawDataSave = rawData;

    trainingData.print();
    updateData(trainingData.rawData);
    krigingModel.updateModelWithNewData(trainingData.rawData);
    kernelRegressionModel.updateData(trainingData.rawData);


	krigingModel.printSurrogateModel();


	rho = 0.0;
	double minimumValidationError = LARGE;
	double best_rho = rho;

#if 1
	printf("Scanning rho between 0.0 and %10.7f\n",numberOfIterForRhoOptimization*dr);
#endif
	for(unsigned int i=0; i<numberOfIterForRhoOptimization; i++){

#if 1
		printf("rho = %10.7f\n",rho);
#endif

		tryModelOnTestSet(testDataForRhoOptimizationLoop);

		double validationError = testDataForRhoOptimizationLoop.calculateMeanSquaredError();
#if 1
		printf("validationError = %10.7f\n",validationError);
#endif

		if(validationError<minimumValidationError){
#if 1
			cout<<"A better rho is found\n";
#endif
			minimumValidationError = validationError;
			best_rho= rho;

		}

		rho += dr;

	}

	rho = best_rho;

	updateData(rawDataSave);
	krigingModel.updateModelWithNewData(rawDataSave);
	kernelRegressionModel.updateData(rawDataSave);

	saveHyperParameters();


}




double AggregationModel::interpolate(rowvec x) const{

	/* find the closest seeding point to the xp in the data set */

	int indx = findNearestNeighbor(x);
	rowvec xNearestPoint = X.row(indx);

#if 0
	cout <<"The closest point to x:\n";
	x.print();
	cout <<"is xg with index = "<<indx<<":\n";
	xNearestPoint.print();
#endif

	rowvec xDiff = x - xNearestPoint;
#if 0
	printVector(xDiff,"xDiff");
#endif
	double min_dist = calculateL1norm(xDiff);
#if 0
	cout<<"min_dist = "<<min_dist<<"\n";
#endif


	double fSurrogateKriging = krigingModel.interpolate(x);
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

	double fSurrogateKernelRegression = kernelRegressionModel.interpolateWithGradients(x);
#if 0
	cout<<"fSurrogateKernelRegression = "<<fSurrogateKernelRegression<<"\n";

#endif

	double w2 = exp(-rho*min_dist*normgrad);
	double w1 = 1.0 - w2;
#if 0
	cout<<"w2 = "<<w2<<"\n";
	cout<<"w1 = "<<w1<<"\n";
	cout<<"result  = "<<w1*fSurrogateKriging+ w2* fSurrogateKernelRegression<<"\n";

#endif

	return w1*fSurrogateKriging+ w2* fSurrogateKernelRegression;


}

double AggregationModel::interpolateWithGradients(rowvec x) const{


	return interpolate(x);

}

void AggregationModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{



}



unsigned int AggregationModel::findNearestNeighbor(rowvec xp) const{

	unsigned int index = -1;
	double minL1Distance = LARGE;



	for(unsigned int i=0; i<X.n_rows; i++){

		rowvec x = X.row(i);

		rowvec xdiff = xp -x;

		double L1distance = calculateL1norm(xdiff);
		if(L1distance< minL1Distance){

			minL1Distance = L1distance;
			index = i;

		}

	}


	return index;


}









