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

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <stack>
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "auxilliary_functions.hpp"
#include "test_functions.hpp"
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"
#include "kernel_regression.hpp"
#include "optimization.hpp"
#include "random_functions.hpp"
#ifdef GPU_VERSION
#include "kernel_regression_cuda.h"
#endif

//#include <codi.hpp>

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
using namespace arma;




TestFunction::TestFunction(std::string name,int dimension){

	if(dimension <= 0){

		fprintf(stderr, "Error: dimension must be at least 1! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);

	}

	numberOfSamplesUsedForVisualization = 10000;
	numberOfInputParams = dimension;
	function_name = name;
	func_ptr = empty;
	adj_ptr  = emptyAdj;
	noiseLevel = 0.0;
	ifFunctionIsNoisy = false;

	lb.zeros(dimension);
	ub.zeros(dimension);


}


void TestFunction::addNoise(double noise){

	noiseLevel = noise;
	ifFunctionIsNoisy = true;

}
void TestFunction::setVisualizationOn(void){

	ifVisualize = true;
}
void TestFunction::setVisualizationOff(void){

	ifVisualize = false;

}



void TestFunction::setBoxConstraints(double lowerBound, double upperBound){

	if(lowerBound >= upperBound){

		fprintf(stderr, "Error: lowerBound cannot be equal or greater than the upperBound! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);

	}
	lb.fill(lowerBound);
	ub.fill(upperBound);


}


void TestFunction::setBoxConstraints(vec lowerBound, vec upperBound){


	for(unsigned int i=0; i<numberOfInputParams; i++){

		if(lowerBound(i) >= upperBound(i)){

			fprintf(stderr, "Error: lowerBound cannot be equal or greater than the upperBound! at %s, line %d.\n",__FILE__, __LINE__);
			exit(-1);

		}
		else{

			lb(i) = lowerBound(i);
			ub(i) = upperBound(i);

		}

	}


}

void TestFunction::evaluateGlobalExtrema(void) const{

	rowvec maxx(numberOfInputParams);
	rowvec minx(numberOfInputParams);

	double globalMin = LARGE;
	double globalMax = -LARGE;

	const int numberOfBruteForceIterations = 100000;

	for(unsigned int i=0; i<numberOfBruteForceIterations; i++ ){

		rowvec x = generateRandomRowVector(lb, ub);
		double functionValue = func_ptr(x.memptr());

		if(functionValue < globalMin){

			globalMin = functionValue;
			minx = x;
		}

		if(functionValue > globalMax){

			globalMax = functionValue;
			maxx = x;
		}

	}

	printf("Brute Force Results:\n");
	printf("Function has minimum at x:\n");
	minx.print();
	printf("Function value at maximum = %10.7f\n",globalMin);
	printf("Function has maximum at x:\n");
	maxx.print();
	printf("Function value at maximum = %10.7f\n",globalMax);

}


void TestFunction::print(void){

	printf("printing function information...\n");
	printf("function name = %s\n",function_name.c_str());
	printf("Number of independent variables = %d\n",numberOfInputParams);
	printf("Parameter bounds:\n");

	for(unsigned int i=0; i<numberOfInputParams; i++){

		printf("x[%d]: %15.10f , %15.10f\n",i,lb(i),ub(i));

	}


}



void TestFunction::plot(int resolution) const{

	if(numberOfInputParams !=2 && numberOfInputParams !=1){

		fprintf(stderr, "Error: only 1D or 2D functions can be visualized! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);

	}

	for(unsigned int j=0; j<numberOfInputParams;j++) {

		if(lb(j) >= ub(j)){

			fprintf(stderr, "Error: Parameter bounds are not set correctly! at %s, line %d.\n",__FILE__, __LINE__);
			exit(-1);

		}
	}

	std::string filename= function_name +"_FunctionPlot.csv";

	if(numberOfInputParams == 1){


		const int resolution = 1000;
		mat visualizationData(resolution,2);


		double dx; /* step sizes in x direction */
		double x;
		double func_val;
		dx = (ub(0)-lb(0))/(resolution-1);

		x = lb(0);
		for(int i=0;i<resolution;i++){


			func_val = func_ptr(&x);

			visualizationData(i,0) = x;
			visualizationData(i,1) = func_val;

			x+= dx;

		}

		visualizationData.save(filename,csv_ascii);


		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function.py "+ function_name;

		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");




	}

	if(numberOfInputParams == 2){


		const int resolution = 100;
		mat visualizationData(resolution*resolution,3);


		double dx,dy; /* step sizes in x and y directions */
		double x[2];
		double func_val;
		dx = (ub(0)-lb(0))/(resolution-1);
		dy = (ub(1)-lb(1))/(resolution-1);

		x[0] = lb(0);
		for(int i=0;i<resolution;i++){
			x[1] = lb(1);
			for(int j=0;j<resolution;j++){
				func_val = func_ptr(x);

				visualizationData(i*resolution+j,0) = x[0];
				visualizationData(i*resolution+j,1) = x[1];
				visualizationData(i*resolution+j,2) = func_val;


				x[1]+=dy;
			}
			x[0]+= dx;

		}


		visualizationData.save(filename,csv_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_2d_surface.py "+ function_name;
		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");



	}


}


double TestFunction::testSurrogateModel(SURROGATE_MODEL modelID, unsigned int howManySamples, bool warmStart){


	printf("Testing surrogate model with the %s function...\n",function_name.c_str());
	printf("ModelId: %d\n",modelID);

	std::string label = function_name;

	std::string filenameSurrogateTest = label+".csv";

	mat sampleMatrix;

	if(modelID == GRADIENT_ENHANCED_KERNEL_REGRESSION || modelID == AGGREGATION) {

		sampleMatrix = generateRandomSamplesWithGradients(howManySamples);
	}
	else{

		sampleMatrix = generateRandomSamples(howManySamples);
	}

	cout << "Saving data file: " <<filenameSurrogateTest<<"\n";
	sampleMatrix.save(filenameSurrogateTest.c_str(), csv_ascii);

	KrigingModel TestFunModelKriging(label);
	KernelRegressionModel TestFunModelKernelRegression(label);
	LinearModel TestFunModelLinearRegression(label);
	AggregationModel TestFunModelAggregation(label);



	switch (modelID) {
	case KRIGING:
	{
		surrogateModel =  &TestFunModelKriging;
		break;
	}
	case LINEAR_REGRESSION:
	{
		surrogateModel = &TestFunModelLinearRegression;
		break;
	}
	case KERNEL_REGRESSION:
	{
		surrogateModel =  &TestFunModelKernelRegression;
		break;
	}
	case GRADIENT_ENHANCED_KERNEL_REGRESSION:
	{
		TestFunModelKernelRegression.setGradientsOn();
		TestFunModelKernelRegression.modelID = GRADIENT_ENHANCED_KERNEL_REGRESSION;
		surrogateModel =  &TestFunModelKernelRegression;
		break;
	}
	case AGGREGATION:
	{

		surrogateModel =  &TestFunModelAggregation;
		break;
	}
	default:
	{
		cout << "ERROR: Non valid modelID\n";
		abort();
	}
	}

	surrogateModel->initializeSurrogateModel();

	if(!warmStart){

		surrogateModel->train();
	}
	else{

		surrogateModel->loadHyperParameters();
	}



	double inSampleError = surrogateModel->calculateInSampleError();
#if 1
	printf("inSampleError = %15.10f\n",inSampleError);
#endif

	mat testSetMatrix = generateRandomSamples(numberOfSamplesUsedForVisualization);

	PartitionData testSet;
	testSet.fillWithData(testSetMatrix);



	surrogateModel->tryModelOnTestSet(testSet);


	label = function_name + "_SurrogateTestResults";

	std::string filenameSurrogateTestResults = label+".csv";

	testSet.saveAsCSVFile(filenameSurrogateTestResults);

	double testError = testSet.calculateMeanSquaredError();
#if 1
	printf("Test Error (MSE)= %15.10f\n",testError);
#endif


	if(ifVisualize){

		surrogateModel->visualizeTestResults();

	}


	return testError;
}




mat TestFunction::generateRandomSamples(unsigned int howManySamples){

	printf("Generating %d random samples for the function: %s\n",howManySamples,function_name.c_str());

	for(unsigned int j=0; j<numberOfInputParams;j++) {

		if (lb(j) >= ub(j)){

			printf("\nERROR: lb(%d) >= ub(%d) (%10.7f >= %10.7f)!\n",j,j,lb(j),ub(j));
			printf("Did you set box constraints properly?\n");
			abort();

		}
	}


	mat sampleMatrix = zeros(howManySamples,numberOfInputParams+1 );

	double *x  = new double[numberOfInputParams];

	for(unsigned int i=0; i<howManySamples; i++ ){

		for(unsigned int j=0; j<numberOfInputParams;j++) {

			x[j] = generateRandomDouble(lb(j), ub(j));

			if(ifFunctionIsNoisy){

				double noiseAdded = noiseLevel*generateRandomDoubleFromNormalDist(-1.0, 1.0, 1.0);
				x[j] += noiseAdded;

			}
		}

		double functionValue = func_ptr(x);

		for(unsigned int j=0;j<numberOfInputParams;j++){

			sampleMatrix(i,j) = x[j];

		}

		sampleMatrix(i,numberOfInputParams) = functionValue;


	}

	delete[] x;

	if(numberOfInputParams == 1){

		sampleMatrix = sort(sampleMatrix);
	}

	return sampleMatrix;


}


mat TestFunction::generateRandomSamplesWithGradients(unsigned int howManySamples){


	printf("Generating %d random samples for the function: %s\n",howManySamples,function_name.c_str());

	for(unsigned int j=0; j<numberOfInputParams;j++) assert(lb(j) < ub(j));

	mat sampleMatrix = zeros(howManySamples,2*numberOfInputParams+1 );

	double *x   = new double[numberOfInputParams];
	double *xb  = new double[numberOfInputParams];


	for(unsigned int i=0; i<howManySamples; i++ ){


		for(unsigned int j=0; j<numberOfInputParams;j++) {

			x[j] = generateRandomDouble(lb(j), ub(j));
		}

		for(unsigned int k=0; k<numberOfInputParams;k++) xb[k] = 0.0;

		double fVal = adj_ptr(x,xb);

		for(unsigned int j=0;j<numberOfInputParams;j++){

			sampleMatrix(i,j) = x[j];

		}

		sampleMatrix(i,numberOfInputParams) = fVal;


		for(unsigned int j=numberOfInputParams+1;j<2*numberOfInputParams+1;j++){

			sampleMatrix(i,j) = xb[j-numberOfInputParams-1];

		}

	}

	delete[] x;
	delete[] xb;

	return sampleMatrix;


}



void TestFunction::testEfficientGlobalOptimization(int nsamplesTrainingData, int maxnsamples, bool ifVisualize, bool ifWarmStart, bool ifMinimize){

	printf("Testing Efficient Global Optimization with the %s function...\n", function_name.c_str());

	std::string filenameEGOTest = function_name+".csv";

	mat samplesMatrix;

	if(!ifWarmStart) {

		samplesMatrix = generateRandomSamples(nsamplesTrainingData);

	}

	std::string problemType;

	if(ifMinimize){

		problemType = "minimize";

	}
	else{
		problemType = "maximize";

	}

	Optimizer OptimizationStudy(function_name,numberOfInputParams,problemType);

	if(ifVisualize) {

		OptimizationStudy.ifVisualize = true;
	}


	std::cout<<"Initializing the objective function..."<<std::endl;

	ObjectiveFunction objFunc(function_name, func_ptr, numberOfInputParams);

	OptimizationStudy.addObjectFunction(objFunc);

	OptimizationStudy.maxNumberOfSamples = maxnsamples;

	OptimizationStudy.setBoxConstraints(lb,ub);

	OptimizationStudy.EfficientGlobalOptimization();

	evaluateGlobalExtrema();



}


double empty(double *x){
	printf("\nERROR: calling empty primal function! Did you set the primal function properly?\n");
	abort();

}

double emptyAdj(double *x, double *xb){
	printf("\nERROR: calling empty dual function! Did you set the dual function properly?\n");
	abort();

	return 0;

}






double Waves2Dpertrubed(double *x){

	double coef = 0.001;
	double term1 = coef*sin(100*x[0]);
	double term2 = coef*cos(100*x[1]);

	if(x[0] > 1 || x[0] < -1 || x[1] > 1 || x[1] < -1){

		return sin(x[0])+ cos(x[1]);

	}
	else{

		return sin(x[0])+ cos(x[1])+term1+term2;
	}

}

double Waves2Dperturbed_adj(double *x,double *xb){
	double coef = 0.001;
	double term1 = coef*sin(100*x[0]);
	double term2 = coef*cos(100*x[1]);

	if(x[0] > 1 || x[0] < -1 || x[1] > 1 || x[1] < -1){
		xb[0] = cos(x[0]);
		xb[1] = -sin(x[1]);
		return sin(x[0])+ cos(x[1]);
	}
	else{

		xb[0] = cos(x[0])+coef*100*cos(100*x[0]);
		xb[1] = -sin(x[1])-coef*100*sin(100*x[1]);

		return sin(x[0])+ cos(x[1])+term1+term2;
	}



}


double Waves2D(double *x){

	return sin(x[0])+ cos(x[1]);

}

double Waves2DAdj(double *x,double *xb){

	xb[0] = cos(x[0]);
	xb[1] = -sin(x[1]);

	return sin(x[0])+ cos(x[1]);




}


double Herbie2D(double *x){

	return exp(-pow((x[0]-1),2))+exp(-0.8*pow((x[0]+1),2))-0.05*sin(8*(x[0]+0.1))
	+ exp(-pow((x[1]-1),2))+exp(-0.8*pow((x[1]+1),2))-0.05*sin(8*(x[1]+0.1));

}

double Herbie2DAdj(double *x, double *xb) {

	xb[0] =  - (2*pow(x[0]-1, 2-1)*exp(-pow(x[0]-1, 2))+2*pow(x[0]+1, 2-1
	)*0.8*exp(-(0.8*pow(x[0]+1, 2)))+8*cos(8*(x[0]+0.1))*0.05);
	xb[1] =  - (2*pow(x[1]-1, 2-1)*exp(-pow(x[1]-1, 2))+2*pow(x[1]+1, 2-1
	)*0.8*exp(-(0.8*pow(x[1]+1, 2)))+8*cos(8*(x[1]+0.1))*0.05);
	return exp(-pow((x[0]-1),2))+exp(-0.8*pow((x[0]+1),2))-0.05*sin(8*(x[0]+0.1))
	+ exp(-pow((x[1]-1),2))+exp(-0.8*pow((x[1]+1),2))-0.05*sin(8*(x[1]+0.1));


}




double test_function1D_adj(double *x, double *xb){


	xb[0] = (2*x[0]+10*(cos(10*x[0])*0.5)+2*cos(2*x[0]));
	return sin(2*x[0])+ 0.5* sin(10*x[0]) + x[0]*x[0] ;

}




double LinearTF1(double *x){

	return 2*x[0]+3*x[1]+1.5;



}

double LinearTF1Adj(double *x, double *xb) {

	xb[0] = xb[0] + 2.0;
	xb[1] = xb[1] + 3.0;
	return 2*x[0]+3*x[1]+1.5;
}



/* Eggholder test function -512<= (x_1,x_2) <= 512
 *
 * global min f(512, 404.2319) = -959.6407
 *
 *
 * */


double Eggholder(double *x){
	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));

}


//double Eggholder_adj(double *xin, double *xb){
//
//	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
//	tape.setActive();
//	codi::RealReverse *x = new codi::RealReverse[2];
//	x[0]= xin[0];
//	x[1]= xin[1];
//	tape.registerInput(x[0]);
//	tape.registerInput(x[1]);
//
//	codi::RealReverse result = -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));
//
//	tape.registerOutput(result);
//
//	tape.setPassive();
//	result.setGradient(1.0);
//	tape.evaluate();
//
//	xb[0]=x[0].getGradient();
//	xb[1]=x[1].getGradient();
//
//#if 0
//	double fdres[2];
//	double epsilon = 0.0001;
//	double xsave;
//	double f0 = Eggholder(xin);
//	xsave = xin[0];
//	xin[0]+=epsilon;
//	double fp = Eggholder(xin);
//	xin[0] = xsave;
//	fdres[0] = (fp-f0)/epsilon;
//	xsave = xin[1];
//	xin[1]+=epsilon;
//	fp = Eggholder(xin);
//	xin[1] = xsave;
//	fdres[1] = (fp-f0)/epsilon;
//	printf("fd results = \n");
//	printf("%10.7f %10.7f\n",fdres[0],fdres[1]);
//
//	printf("ad results = \n");
//	printf("%10.7f %10.7f\n",xb[0],xb[1]);
//#endif
//	delete[] x;
//	tape.reset();
//	return result.getValue();
//
//}



/* McCormick test function -1.5<= x_1 <= 4  -3<= x_2 <= 4
 *
 * global min f(-0.54719, -1.54719) = -1.9133
 *
 *
 * */

double McCormick(double *x){

	return sin(x[0]+x[1])+ pow( (x[0]-x[1]),2.0)-1.5*x[0]+2.5*x[1]+1.0;

}

double McCormick_adj(double *x, double *xb) {
	double tempb;
	double tempb0;

	tempb = cos(x[0]+x[1]);
	tempb0 = 2.0*pow(x[0]-x[1], 2.0-1);
	xb[0] = xb[0] + tempb0 - 1.5 + tempb;
	xb[1] = xb[1] + 2.5 - tempb0 + tempb;

	return sin(x[0]+x[1])+ pow( (x[0]-x[1]),2.0)-1.5*x[0]+2.5*x[1]+1.0;
}




/* Goldstein-Price test function
 *
 *
 * global min f(0, -1) = 3
 *
 *
 * */

double Goldstein_Price(double *x){
	double x_sqr  =  x[0]*x[0]; // x^2
	double y_sqr  =  x[1]*x[1]; // y^2
	double xy     =  x[0]*x[1]; // xy
	double temp1  = pow( (x[0]+x[1]+1), 2.0);
	double temp2  = (19.0-14.0*x[0]+3.0*x_sqr-14.0*x[1]+6.0*xy+ 3.0*y_sqr);
	double temp3  = pow( (2*x[0]-3.0*x[1]), 2.0);
	double temp4  = (18.0 - 32.0 *x[0] + 12.0* x_sqr +48.0*x[1]-36.0*xy+27.0*y_sqr);

	return (1.0+temp1*temp2)*(30.0+temp3*temp4);

}


double Goldstein_Price_adj(double *x, double *xb) {
	double x_sqr = x[0]*x[0];
	double x_sqrb;
	// x^2
	double y_sqr = x[1]*x[1];
	double y_sqrb;
	// y^2
	double xy = x[0]*x[1];
	double xyb;
	// xy
	double temp1;
	double temp1b;
	double tempb;
	double tempb0;
	double tempb1;
	double tempb2;
	temp1 = pow(x[0] + x[1] + 1, 2.0);
	double temp2 = 19.0 - 14.0*x[0] + 3.0*x_sqr - 14.0*x[1] + 6.0*xy + 3.0*
			y_sqr;
	double temp2b;
	double temp3;
	double temp3b;
	temp3 = pow(2*x[0] - 3.0*x[1], 2.0);
	double temp4 = 18.0 - 32.0*x[0] + 12.0*x_sqr + 48.0*x[1] - 36.0*xy + 27.0*
			y_sqr;
	double temp4b;
	tempb = (temp3*temp4+30.0);
	tempb0 = (temp1*temp2+1.0);
	temp1b = temp2*tempb;
	temp2b = temp1*tempb;
	temp3b = temp4*tempb0;
	temp4b = temp3*tempb0;
	x_sqrb = 3.0*temp2b + 12.0*temp4b;
	xb[0] = xb[0] - 32.0*temp4b;
	xb[1] = xb[1] + 48.0*temp4b;
	y_sqrb = 3.0*temp2b + 27.0*temp4b;
	xyb = 6.0*temp2b - 36.0*temp4b;
	tempb1 = 2.0*pow(2*x[0]-3.0*x[1], 2.0-1)*temp3b;
	xb[0] = xb[0] + 2*tempb1;
	xb[1] = xb[1] - 3.0*tempb1;
	xb[0] = xb[0] - 14.0*temp2b;
	xb[1] = xb[1] - 14.0*temp2b;
	tempb2 = 2.0*pow(x[0]+x[1]+1, 2.0-1)*temp1b;
	xb[0] = xb[0] + tempb2;
	xb[1] = xb[1] + tempb2;
	xb[0] = xb[0] + x[1]*xyb;
	xb[1] = xb[1] + x[0]*xyb;
	xb[1] = xb[1] + 2*x[1]*y_sqrb;
	xb[0] = xb[0] + 2*x[0]*x_sqrb;


	return (1.0+temp1*temp2)*(30.0+temp3*temp4);
}


/* Rosenbrock test function with a = 1 and b= 100
 *
 * f(x,y) = (a-x)^2 + b(y-x^2)^2
 * global min f(x,y) = (a,a^2)
 *
 *
 * */

double Rosenbrock(double *x){

	return (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);


}

double Rosenbrock_adj(double *x, double *xb) {
	double tempb;
	double Rosenbrockb = 1.0;

	xb[0] = 0.0;
	xb[1] = 0.0;
	tempb = 100.0*2*(x[1]-x[0]*x[0])*Rosenbrockb;
	xb[0] = xb[0] - 2*x[0]*tempb - 2*(1.0-x[0])*Rosenbrockb;
	xb[1] = xb[1] + tempb;

	return ( (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]));

}


double Rosenbrock3D(double *x){

	double term1 = (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	double term2 = (1.0-x[1])* (1.0-x[1]) + 100.0 *(x[2]-x[1]*x[1])*(x[2]-x[1]*x[1]);
	return term1+term2;


}

double Rosenbrock3D_adj(double *x, double *xb) {

	double Rosenbrock3Db = 1.0;
	xb[0] = 0.0; xb[1] = 0.0; xb[2] = 0.0;

	double term1 = (1.0-x[0])*(1.0-x[0]) + 100.0*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	double term1b = 0.0;
	double term2 = (1.0-x[1])*(1.0-x[1]) + 100.0*(x[2]-x[1]*x[1])*(x[2]-x[1]*x[1]);
	double term2b = 0.0;
	double tempb;
	double tempb0;

	term1b = Rosenbrock3Db;
	term2b = Rosenbrock3Db;
	tempb = 100.0*2*(x[2]-x[1]*x[1])*term2b;
	xb[1] = xb[1] - 2*x[1]*tempb - 2*(1.0-x[1])*term2b;
	xb[2] = xb[2] + tempb;
	tempb0 = 100.0*2*(x[1]-x[0]*x[0])*term1b;
	xb[0] = xb[0] - 2*x[0]*tempb0 - 2*(1.0-x[0])*term1b;
	xb[1] = xb[1] + tempb0;
	return term1+term2;
}

double Rosenbrock4D(double *x){

	double term1 = (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	double term2 = (1.0-x[1])* (1.0-x[1]) + 100.0 *(x[2]-x[1]*x[1])*(x[2]-x[1]*x[1]);
	double term3 = (1.0-x[2])* (1.0-x[2]) + 100.0 *(x[3]-x[2]*x[2])*(x[3]-x[2]*x[2]);
	return term1+term2+term3;


}

double Rosenbrock4D_adj(double *x, double *xb) {
	double Rosenbrock4Db = 1.0;
	xb[0] = 0.0; xb[1] = 0.0; xb[2] = 0.0; xb[3] = 0.0;
	double term1 = (1.0-x[0])*(1.0-x[0]) + 100.0*(x[1]-x[0]*x[0])*(x[1]-x[0]*x
			[0]);
	double term1b = 0.0;
	double term2 = (1.0-x[1])*(1.0-x[1]) + 100.0*(x[2]-x[1]*x[1])*(x[2]-x[1]*x
			[1]);
	double term2b = 0.0;
	double term3 = (1.0-x[2])*(1.0-x[2]) + 100.0*(x[3]-x[2]*x[2])*(x[3]-x[2]*x
			[2]);
	double term3b = 0.0;
	double tempb;
	double tempb0;
	double tempb1;
	term1b = Rosenbrock4Db;
	term2b = Rosenbrock4Db;
	term3b = Rosenbrock4Db;
	tempb = 100.0*2*(x[3]-x[2]*x[2])*term3b;
	xb[2] = xb[2] - 2*x[2]*tempb - 2*(1.0-x[2])*term3b;
	xb[3] = xb[3] + tempb;
	tempb0 = 100.0*2*(x[2]-x[1]*x[1])*term2b;
	xb[1] = xb[1] - 2*x[1]*tempb0 - 2*(1.0-x[1])*term2b;
	xb[2] = xb[2] + tempb0;
	tempb1 = 100.0*2*(x[1]-x[0]*x[0])*term1b;
	xb[0] = xb[0] - 2*x[0]*tempb1 - 2*(1.0-x[0])*term1b;
	xb[1] = xb[1] + tempb1;
	return term1+term2+term3;
}

double Rosenbrock8D(double *x){


	double temp=0.0;

	for(int i=0; i<7; i++){

		temp+= 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1-x[i])*(1-x[i]);
	}

	return temp;

}


double Rosenbrock8D_adj(double *x, double *xb) {
	double temp = 0.0;
	double tempb0 = 0.0;


	for(int i=0; i<8; i++){
		xb[i]=0.0;

	}

	for(int i=0; i<7; i++){

		temp+= 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1-x[i])*(1-x[i]);
	}


	tempb0 = 1.0;
	{
		double tempb;
		for (int i = 6; i > -1; --i) {
			tempb = 100.0*2*(x[i+1]-x[i]*x[i])*tempb0;
			xb[i + 1] = xb[i + 1] + tempb;
			xb[i] = xb[i] - 2*(1-x[i])*tempb0 - 2*x[i]*tempb;
		}
	}

	return temp;
}


double Shubert(double *x){

	double term1 =0.0;
	double term2 =0.0;

	for(int i=0; i<5; i++){

		term1+= i*cos( (i+1)*x[0]+i );

	}

	for(int j=0; j<5; j++){

		term2+= j*cos((j+1)*x[1]+j);

	}


	return term1*term2;

}


double Shubert_adj(double *x, double *xb) {
	double term1 = 0.0;
	double term1b = 0.0;
	double term2 = 0.0;
	double term2b = 0.0;
	double Shubertb=1.0;

	xb[0]=0.0;
	xb[1]=0.0;

	for (int i = 0; i < 5; ++i)
		term1 = term1 + i*cos((i+1)*x[0]+i);
	for (int j = 0; j < 5; ++j)
		term2 = term2 + j*cos((j+1)*x[1]+j);
	term1b = term2*Shubertb;
	term2b = term1*Shubertb;
	for (int j = 4; j > -1; --j)
		xb[1] = xb[1] - sin(j+(j+1)*x[1])*j*(j+1)*term2b;
	for (int i = 4; i > -1; --i)
		xb[0] = xb[0] - sin(i+(i+1)*x[0])*i*(i+1)*term1b;

	return term1*term2;

}



/* Himmelblau test function
 *
 * f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
 * four local minima :
 * f(3.0,2.0)=0.0
 * f(2.805118, 3.131312) = 0.0
 * f(-3.779310, -3.283186) = 0.0
 * f(3.584428, -1.848126)  = 0.0
 *
 *
 * */

double Himmelblau(double *x){

	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );


}

double HimmelblauAdj(double *x, double *xb) {
	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	xb[0] = tempb0 + 2*x[0]*tempb;
	xb[1] = 2*x[1]*tempb0 + tempb;

	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );

}


/*
 *
 * The Borehole function models water flow through a borehole.
 * Its simplicity and quick evaluation makes it a commonly used function
 * for testing a wide variety of methods in computer experiments.
 * The response is water flow rate, in m3/yr.
 *
 *
 * f(x) = 2*pi* Tu (Hu-Hl) / ( ln(r/rw)* ( 1+ 2* L* Tu/( ln(r/rw)*rw^2* Kw ) + Tu/Tl ) )
 * rw (x[0]) : radius of borehole (m) [0.05, 0.15]
 * r   (x[1]) : radius of influence(m) [100, 50000]
 * Tu (x[2]) : transmissivity of upper aquifer (m2/yr) [63500, 115600]
 * Hu (x[3]) : potentiometric head of upper aquifer (m)  [990, 1110]
 * Tl (x[4]) : transmissivity of lower aquifer (m2/yr) [63.1, 116]
 * Hl (x[5]) : potentiometric head of lower aquifer (m) [700, 820]
 * L   (x[6]) : length of borehole (m)   [1120, 1680]
 * Kw (x[7]) : hydraulic conductivity of borehole (m/yr) [9855, 12045]
 */


double Borehole(double *x){

	double pi = 3.14159265359;

	double ln_r_div_rw = log(x[1]/x[0]);  // ln(r/rw)
	double num = 2*pi* x[2]*(x[3]- x[5]); // 2pi*Tu*(Hu-Hl)
	double two_L_Tu = 2.0* x[6]*x[2];
	double den = ln_r_div_rw * (1.0 + two_L_Tu /( ( ln_r_div_rw )* x[0]*x[0]*x[7])+ x[2]/x[4] );
	return num/den;

}




/*
 *  Sw: Wing Area (ft^2) (150,200)
 *  Wfw: Weight of fuel in the wing (lb) (220,300)
 *  A: Aspect ratio (6,10)
 *  Lambda: quarter chord sweep (deg) (-10,10)
 *  q: dynamic pressure at cruise (lb/ft^2)  (16,45)
 *  lambda: taper ratio (0.5,1)
 *  tc: aerofoil thickness to chord ratio (0.08,0.18)
 *  Nz: ultimate load factor (2.5,6)
 *  Wdg: flight design gross weight (lb)  (1700,2500)
 *  Wp: paint weight (lb/ft^2) (0.025, 0.08)
 *
 */
double Wingweight(double *x){
#if 0
	printf("x = \n");
	for(int i=0; i<10; i++) {

		printf("%10.7f ",x[i]);

	}
	printf("\n");
#endif

	double Sw=x[0];
	double Wfw=x[1];
	double A=x[2];
	double Lambda=x[3];
	double q=x[4];
	double lambda=x[5];
	double tc=x[6];
	double Nz=x[7];
	double Wdg=x[8];
	double Wp=x[9];


	double deg = (Lambda*datum::pi)/180.0;

	double W = 0.036*pow(Sw,0.758)*pow(Wfw,0.0035)*pow((A/(cos(deg)*cos(deg))),0.6) *
			pow(q,0.006)*pow(lambda,0.04)*pow( (100.0*tc/cos(deg)), -0.3) *pow( (Nz*Wdg),0.49) + Sw*Wp;

	return(W);
}

double WingweightAdj(double *x, double *xb) {
	double Sw = x[0];
	double Swb = 0.0;
	double Wfw = x[1];
	double Wfwb = 0.0;
	double A = x[2];
	double Ab = 0.0;
	double Lambda = x[3];
	double Lambdab = 0.0;
	double q = x[4];
	double qb = 0.0;
	double lambda = x[5];
	double lambdab = 0.0;
	double tc = x[6];
	double tcb = 0.0;
	double Nz = x[7];
	double Nzb = 0.0;
	double Wdg = x[8];
	double Wdgb = 0.0;
	double Wp = x[9];
	double Wpb = 0.0;
	double deg = Lambda*datum::pi/180.0;
	double degb = 0.0;
	double W = 0.036*pow(Sw, 0.758)*pow(Wfw, 0.0035)*pow(A/(cos(deg)*cos(deg))
			, 0.6)*pow(q, 0.006)*pow(lambda, 0.04)*pow(100.0*tc/cos(deg), -0.3)*pow(Nz
					*Wdg, 0.49) + Sw*Wp;
	double Wb = 0.0;
	double temp;
	double temp0;
	double temp1;
	double temp2;
	double temp3;
	double temp4;
	double temp5;
	double tempb;
	double temp6;
	double temp7;
	double temp8;
	double tempb0;
	double temp9;
	double temp10;
	double temp11;
	double temp12;
	double tempb1;
	double tempb2;
	double tempb3;
	double tempb4;

	Wb = 1.0;
	temp = pow(lambda, 0.04);
	temp0 = pow(q, 0.006);
	temp1 = temp0*temp;
	temp2 = cos(deg);
	temp3 = temp2*temp2;
	temp4 = A/temp3;
	temp5 = pow(temp4, 0.6);
	temp6 = cos(deg);
	temp7 = tc/temp6;
	temp8 = pow(100.0*temp7, -0.3);
	temp9 = pow(Nz*Wdg, 0.49);
	temp10 = pow(Wfw, 0.0035);
	temp11 = pow(Sw, 0.758);
	temp12 = temp11*temp10;
	tempb = temp5*temp1*0.036*Wb;
	tempb3 = temp12*temp9*temp8*0.036*Wb;
	Wpb = Sw*Wb;
	tempb4 = 0.6*pow(temp4, 0.6-1)*temp1*tempb3/temp3;
	qb = 0.006*pow(q, 0.006-1)*temp*temp5*tempb3;
	lambdab = 0.04*pow(lambda, 0.04-1)*temp0*temp5*tempb3;
	Ab = tempb4;
	tempb0 = temp8*tempb;
	Swb = Wp*Wb + 0.758*pow(Sw, 0.758-1)*temp10*temp9*tempb0;
	tempb2 = -(100.0*0.3*pow(100.0*temp7, -0.3-1)*temp12*temp9*tempb/temp6);
	degb = sin(deg)*2*temp2*temp4*tempb4 + sin(deg)*temp7*tempb2;
	tcb = tempb2;
	Wfwb = 0.0035*pow(Wfw, 0.0035-1)*temp11*temp9*tempb0;
	tempb1 = 0.49*pow(Nz*Wdg, 0.49-1)*temp12*tempb0;
	Nzb = Wdg*tempb1;
	Wdgb = Nz*tempb1;
	Lambdab = datum::pi*degb/180.0;
	xb[9] = Wpb;
	xb[8] = Wdgb;
	xb[7] = Nzb;
	xb[6] = tcb;
	xb[5] = lambdab;
	xb[4] = qb;
	xb[3] = Lambdab;
	xb[2] = Ab;
	xb[1] = Wfwb;
	xb[0] = Swb;
	return(W);
}
















