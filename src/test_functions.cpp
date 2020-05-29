
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

#include <codi.hpp>

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
using namespace arma;




TestFunction::TestFunction(std::string name,int dimension){

	if(dimension <= 0){

		fprintf(stderr, "Error: dimension must be at least 1! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);

	}
	numberOfInputParams = dimension;
	function_name = name;
	func_ptr = empty;
	adj_ptr  = emptyAdj;
	noiseLevel = 0.0;
	ifFunctionIsNoisy = false;
	ifAdjointFunctionExist = false;
	lb.zeros(dimension);
	ub.zeros(dimension);


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

void TestFunction::testSurrogateModel(SURROGATE_MODEL modelID, unsigned int howManySamples, bool ifVisualize){

	printf("Testing surrogate model with the %s function...\n", function_name.c_str());
	printf("ModelId: %d\n",modelID);

	std::string label = function_name;

	std::string filenameSurrogateTest = label+".csv";

	mat sampleMatrix;

	if(modelID == AGGREGATION) {
		sampleMatrix = generateRandomSamplesWithGradients(howManySamples);
	}
	else{

		sampleMatrix = generateRandomSamples(howManySamples);
	}


	sampleMatrix.save(filenameSurrogateTest.c_str(), csv_ascii);



	SurrogateModel *TestFunSurrogate;


	KrigingModel TestFunModelKriging;
	LinearModel TestFunModelLinearRegression;

	if(modelID == KRIGING){

		TestFunModelKriging = KrigingModel(label,numberOfInputParams);
		TestFunSurrogate = &TestFunModelKriging;


	}

	if(modelID == LINEAR_REGRESSION){

		TestFunModelLinearRegression = LinearModel(label,numberOfInputParams);
		TestFunSurrogate = &TestFunModelLinearRegression;


	}


	TestFunSurrogate->print();

	TestFunSurrogate->train();


	double inSampleError = TestFunSurrogate->calculateInSampleError();
#if 1
	printf("inSampleError = %15.10f\n",inSampleError);
#endif

	mat testSet = generateRandomSamples(20000);
	mat testResults = TestFunSurrogate->tryModelOnTestSet(testSet);

	label = function_name + "_SurrogateTestResults";

	std::string filenameSurrogateTestResults = label+".csv";


	testResults.save(filenameSurrogateTestResults.c_str(), csv_ascii);

	double testError = mean(testResults.col(numberOfInputParams+2));
#if 1
	printf("Test Error (MSE)= %15.10f\n",testError);
#endif


	if(ifVisualize){

		TestFunSurrogate->visualizeTestResults(testResults);

	}


}





mat TestFunction::generateRandomSamples(unsigned int howManySamples){

	printf("Generating %d random samples for the function: %s\n",howManySamples,function_name.c_str());

	for(unsigned int j=0; j<numberOfInputParams;j++) assert(lb(j) < ub(j));


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


	return sampleMatrix;


}


mat TestFunction::generateRandomSamplesWithGradients(unsigned int howManySamples){

	assert(ifAdjointFunctionExist);

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
	fprintf(stderr, "Error: calling empty function! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);
	return 0;

}

double emptyAdj(double *x, double *xb){
	fprintf(stderr, "Error: calling emptyAdj function! at %s, line %d.\n",__FILE__, __LINE__);
	exit(-1);

	return 0;

}


double test_function1KernelReg(double *x){

	return 0.001*sin(x[0])+ 1.0*cos(x[1]);

}

double test_function1KernelRegAdj(double *xin, double *xb){

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[2];
	x[0]= xin[0];
	x[1]= xin[1];
	tape.registerInput(x[0]);
	tape.registerInput(x[1]);

	codi::RealReverse result = 0.001*sin(x[0])+ 1.0*cos(x[1]);

	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	xb[0]=x[0].getGradient();
	xb[1]=x[1].getGradient();

	tape.reset();
	delete[] x;
	return result.getValue();

}


double test_function2KernelReg(double *x){

	return sin(x[1]+x[2])+ pow( (x[1]-x[2]),2.0)-1.5*x[1]+2.5*x[2]+1.0;

}

double test_function2KernelRegAdj(double *xin,double *xb){
	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[5];

	for(int i=0; i<5;i++){

		x[i]= xin[i];
		tape.registerInput(x[i]);
	}
	codi::RealReverse result = sin(x[1]+x[2])+ pow( (x[1]-x[2]),2.0)-1.5*x[1]+2.5*x[2]+1.0;
	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<5;i++){
		xb[i]=x[i].getGradient();

	}
	delete[] x;
	tape.reset();
	return result.getValue();

}



double TestFunction1D(double *x){

	return sin(2*x[0])+ 0.5* sin(10*x[0]) + x[0]*x[0] ;

}

double TestFunction1DAdj(double *xin, double *xb){

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[1];

	tape.registerInput(x[0]);

	codi::RealReverse result = sin(2*x[0])+ 0.5* sin(10*x[0]) + x[0]*x[0] ;
	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();


	xb[0]=x[0].getGradient();


	delete[] x;
	tape.reset();
	return result.getValue();

}




double Griewank2D(double *x){

	return (x[0]*x[0])/4000.0 + (x[1]*x[1])/4000.0 - cos(x[0]) * cos(x[1]/sqrt(2.0));



}

double Griewank2D_adj(double *xin, double *xb){

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[2];

	for(int i=0; i<2;i++){

		x[i]= xin[i];
		tape.registerInput(x[i]);
	}

	codi::RealReverse result = (x[0]*x[0])/4000.0 + (x[1]*x[1])/4000.0 - cos(x[0]) * cos(x[1]/sqrt(2.0));
	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<2;i++){
		xb[i]=x[i].getGradient();

	}
	delete[] x;
	tape.reset();
	return result.getValue();





}

double Rastrigin6D(double *x){

	int d = 6;
	double sum = 0;
	for (int i = 0; i<d; i++){

		sum = sum + (x[i]*x[i] - 10.0*cos(2*3.14159265359*x[i]));
	}

	sum = 10.0*d + sum;
	return sum;


}

double Rastrigin6D_adj(double *xin, double *xb){

	int d = 6;

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[d];

	for(int i=0; i<d;i++){

		x[i]= xin[i];
		tape.registerInput(x[i]);
	}


	codi::RealReverse sum = 0;
	for (int i = 0; i<d; i++){

		sum = sum + (x[i]*x[i] - 10.0*cos(2*3.14159265359*x[i]));
	}

	sum = 10.0*d + sum;


	tape.registerOutput(sum);

	tape.setPassive();
	sum.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<d;i++){
		xb[i]=x[i].getGradient();

	}
	delete[] x;
	tape.reset();
	return sum.getValue();





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

double Waves2D_adj(double *x,double *xb){

	xb[0] = cos(x[0]);
	xb[1] = -sin(x[1]);

	return sin(x[0])+ cos(x[1]);




}


double Herbie2D(double *x){

	return exp(-pow((x[0]-1),2))+exp(-0.8*pow((x[0]+1),2))-0.05*sin(8*(x[0]+0.1))
	+ exp(-pow((x[1]-1),2))+exp(-0.8*pow((x[1]+1),2))-0.05*sin(8*(x[1]+0.1));

}

double Herbie2D_adj(double *xin, double *xb) {

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[2];

	for(int i=0; i<2;i++){

		x[i]= xin[i];
		tape.registerInput(x[i]);
	}


	codi::RealReverse result = exp(-pow((x[0]-1),2))+exp(-0.8*pow((x[0]+1),2))-0.05*sin(8*(x[0]+0.1))
	+ exp(-pow((x[1]-1),2))+exp(-0.8*pow((x[1]+1),2))-0.05*sin(8*(x[1]+0.1));
	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<2;i++){
		xb[i]=x[i].getGradient();

	}
	delete[] x;
	tape.reset();
	return result.getValue();

}



double test_function1D_adj(double *x, double *xb){


	xb[0] = (2*x[0]+10*(cos(10*x[0])*0.5)+2*cos(2*x[0]));
	return sin(2*x[0])+ 0.5* sin(10*x[0]) + x[0]*x[0] ;

}




double simple_2D_linear_test_function1(double *x){

	return 2*x[0]+3*x[1]+1.5;



}

double simple_2D_linear_test_function1_adj(double *x, double *xb) {

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


double Eggholder_adj(double *xin, double *xb){

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[2];
	x[0]= xin[0];
	x[1]= xin[1];
	tape.registerInput(x[0]);
	tape.registerInput(x[1]);

	codi::RealReverse result = -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));

	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	xb[0]=x[0].getGradient();
	xb[1]=x[1].getGradient();

#if 0
	double fdres[2];
	double epsilon = 0.0001;
	double xsave;
	double f0 = Eggholder(xin);
	xsave = xin[0];
	xin[0]+=epsilon;
	double fp = Eggholder(xin);
	xin[0] = xsave;
	fdres[0] = (fp-f0)/epsilon;
	xsave = xin[1];
	xin[1]+=epsilon;
	fp = Eggholder(xin);
	xin[1] = xsave;
	fdres[1] = (fp-f0)/epsilon;
	printf("fd results = \n");
	printf("%10.7f %10.7f\n",fdres[0],fdres[1]);

	printf("ad results = \n");
	printf("%10.7f %10.7f\n",xb[0],xb[1]);
#endif
	delete[] x;
	tape.reset();
	return result.getValue();

}



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

double Himmelblau_adj(double *x, double *xb) {
	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	xb[0] = xb[0] + tempb0 + 2*x[0]*tempb;
	xb[1] = xb[1] + 2*x[1]*tempb0 + tempb;

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


double BoreholeAdj(double *xin, double *xb){


	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[10];

	for(int i=0; i<10; i++) {

		x[i] = xin[i];
		tape.registerInput(x[i]);
	}

	codi::RealReverse ln_r_div_rw = log(x[1]/x[0]);  // ln(r/rw)
	codi::RealReverse num = 2.0*datum::pi* x[2]*(x[3]- x[5]); // 2pi*Tu*(Hu-Hl)
	codi::RealReverse two_L_Tu = 2.0* x[6]*x[2];
	codi::RealReverse den = ln_r_div_rw * (1.0 + two_L_Tu /( ( ln_r_div_rw )* x[0]*x[0]*x[7])+ x[2]/x[4] );
	codi::RealReverse result = num/den;

	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<8; i++) {

		xb[i]=x[i].getGradient();
	}


#if 0
	double fdres[8];
	double epsilon = 0.0;
	double xsave;
	double fp = 0.0;

	double f0 = Borehole(xin);
	for(int i=0; i<8; i++){

		epsilon = xin[i]*0.001;
		xsave = xin[i];
		//		printf("xin[%d] = %20.15f\n",i,xin[i]);
		xin[i]+=epsilon;
		//		printf("xin[%d] = %20.15f\n",i,xin[i]);
		fp =  Wingweight(xin);
		xin[i] = xsave;
		printf("fp = %20.15f f0 = %20.15f\n",fp,f0);

		fdres[i] = (fp-f0)/epsilon;
	}

	printf("fd results = \n");
	for(int i=0; i<10; i++) {
		printf("%10.7f ",fdres[i]);
	}
	printf("\n");
	printf("ad results = \n");
	for(int i=0; i<10; i++) {
		printf("%10.7f ",xb[i]);
	}
	printf("\n");

#endif
	delete[] x;
	tape.reset();

	return result.getValue();



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

double WingweightAdj(double *xin, double *xb){


	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();
	codi::RealReverse *x = new codi::RealReverse[10];

	for(int i=0; i<10; i++) {

		x[i] = xin[i];
		tape.registerInput(x[i]);
	}


	codi::RealReverse Sw=x[0];
	codi::RealReverse Wfw=x[1];
	codi::RealReverse A=x[2];
	codi::RealReverse Lambda=x[3];
	codi::RealReverse q=x[4];
	codi::RealReverse lambda=x[5];
	codi::RealReverse tc=x[6];
	codi::RealReverse Nz=x[7];
	codi::RealReverse Wdg=x[8];
	codi::RealReverse Wp=x[9];


	codi::RealReverse deg = (Lambda*datum::pi)/180.0;

	codi::RealReverse result = 0.036*pow(Sw,0.758)*pow(Wfw,0.0035)*pow((A/(cos(deg)*cos(deg))),0.6) *
			pow(q,0.006)*pow(lambda,0.04)*pow( (100.0*tc/cos(deg)), -0.3) *pow( (Nz*Wdg),0.49) + Sw*Wp;

	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();

	for(int i=0; i<10; i++) {

		xb[i]=x[i].getGradient();
	}


#if 0
	double fdres[10];
	double epsilon = 0.0;
	double xsave;
	double fp = 0.0;

	double f0 = Wingweight(xin);
	for(int i=0; i<10; i++){

		epsilon = xin[i]*0.001;
		xsave = xin[i];
		//		printf("xin[%d] = %20.15f\n",i,xin[i]);
		xin[i]+=epsilon;
		//		printf("xin[%d] = %20.15f\n",i,xin[i]);
		fp =  Wingweight(xin);
		xin[i] = xsave;
		printf("fp = %20.15f f0 = %20.15f\n",fp,f0);

		fdres[i] = (fp-f0)/epsilon;
	}

	printf("fd results = \n");
	for(int i=0; i<10; i++) {
		printf("%10.7f ",fdres[i]);
	}
	printf("\n");
	printf("ad results = \n");
	for(int i=0; i<10; i++) {
		printf("%10.7f ",xb[i]);
	}
	printf("\n");

#endif
	delete[] x;
	tape.reset();

	return result.getValue();



}




/*
 *
 * Generate test data for a given 2D function with random distribution
 *
 */


/* generate the contour plot with the given function within specified bounds and resolution */
void generate_contour_plot_2D_function(double (*test_function)(double *), double *bounds, std::string function_name, int resolution=100){
	/* first generate output data file */

	std::string filename= function_name;
	filename += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".dat";


	std::string file_name_for_plot = function_name;
	file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";



	printf("opening file %s for output...\n",filename.c_str());
	FILE *test_function_data=fopen(filename.c_str(),"w");



	double dx,dy; /* step sizes in x and y directions */
	double x[2];
	double func_val;
	dx = (bounds[1]-bounds[0])/(resolution-1);
	dy = (bounds[3]-bounds[2])/(resolution-1);

	x[0] = bounds[0];
	for(int i=0;i<resolution;i++){
		x[1] = bounds[2];
		for(int j=0;j<resolution;j++){
			func_val = test_function(x);
			fprintf(test_function_data,"%10.7f %10.7f %10.7f\n",x[0],x[1],func_val);


			x[1]+=dy;
		}
		x[0]+= dx;

	}

	fclose(test_function_data);

	std::string python_command = "python -W ignore plot_2d_surface.py "+ filename+ " "+ file_name_for_plot ;



	FILE* in = popen(python_command.c_str(), "r");


	fprintf(in, "\n");



}


/* generate the contour plot with the given function
 * and its gradient sensitivities within specified bounds and resolution
 *
 *   */
void generate_contour_plot_2D_function_with_gradient(double (*test_function)(double *, double *),
		double *bounds,
		std::string function_name,
		std::string python_dir,
		int resolution=100){
	/* first generate output data file */

	std::string filename= function_name;
	filename += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".dat";

	std::string filename_g1= function_name;
	filename_g1 += "_g1_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".dat";


	std::string filename_g2= function_name;
	filename_g2 += "_g2_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".dat";



	std::string file_name_for_plot = function_name;
	file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";


	std::string file_name_for_plot_g1 = function_name;
	file_name_for_plot_g1 += "_g1_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

	std::string file_name_for_plot_g2 = function_name;
	file_name_for_plot_g2 += "_g2_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";




	printf("opening file %s for output...\n",filename.c_str());



	FILE *test_function_data=fopen(filename.c_str(),"w");
	FILE *test_function_data_g1=fopen(filename_g1.c_str(),"w");
	FILE *test_function_data_g2=fopen(filename_g2.c_str(),"w");



	double dx,dy; /* step sizes in x and y directions */
	double x[2];
	double xb[2];
	double func_val;
	dx = (bounds[1]-bounds[0])/(resolution-1);
	dy = (bounds[3]-bounds[2])/(resolution-1);

	x[0] = bounds[0];
	for(int i=0;i<resolution;i++){
		x[1] = bounds[2];
		for(int j=0;j<resolution;j++){
			xb[0]=0.0; xb[1]=0.0;
			func_val = test_function(x,xb);
			fprintf(test_function_data,"%10.7f %10.7f %10.7f\n",x[0],x[1],func_val);
			fprintf(test_function_data_g1,"%10.7f %10.7f %10.7f\n",x[0],x[1],xb[0]);
			fprintf(test_function_data_g2,"%10.7f %10.7f %10.7f\n",x[0],x[1],xb[1]);

			x[1]+=dy;
		}
		x[0]+= dx;

	}

	fclose(test_function_data);
	fclose(test_function_data_g1);
	fclose(test_function_data_g2);

	std::string title = "function";
	std::string python_command = "python -W ignore "+python_dir+"/plot_2d_surface.py "+ filename+ " "+ file_name_for_plot + " "+title;


	FILE* in = popen(python_command.c_str(), "r");
	fprintf(in, "\n");

	title = "df/dx1";
	std::string python_command_g1 = "python -W ignore "+python_dir+"/plot_2d_surface.py "+ filename_g1+ " "+ file_name_for_plot_g1+ " "+title ;
	FILE* in_g1 = popen(python_command_g1.c_str(), "r");
	fprintf(in_g1, "\n");

	title = "df/dx2";
	std::string python_command_g2 = "python -W ignore "+python_dir+"/plot_2d_surface.py "+ filename_g2+ " "+ file_name_for_plot_g2 + " "+title;
	FILE* in_g2 = popen(python_command_g2.c_str(), "r");
	fprintf(in_g2, "\n");




}



void generate_plot_1D_function(double (*test_function)(double *), double *bounds, std::string function_name, int resolution=1000){
	/* first generate output data file */

	std::string filename= function_name;
	filename += "_"+std::to_string(resolution)+".dat";


	std::string file_name_for_plot = function_name;
	file_name_for_plot += "_"+std::to_string(resolution)+".png";



	printf("opening file %s for output...\n",filename.c_str());
	FILE *test_function_data=fopen(filename.c_str(),"w");



	double dx; /* step size in x*/
	double x;
	double func_val;
	dx = (bounds[1]-bounds[0])/(resolution-1);


	x= bounds[0];
	for(int i=0;i<resolution;i++){

		func_val = test_function(&x);
		fprintf(test_function_data,"%10.7f %10.7f\n",x,func_val);
		x+= dx;

	}




	fclose(test_function_data);

	std::string python_command = "python -W ignore plot_1d_function.py "+ filename+ " "+ file_name_for_plot ;



	FILE* in = popen(python_command.c_str(), "r");


	fprintf(in, "\n");



}





//void perform_NNregression_test(double (*test_function)(double *),
//		double *bounds,
//		std::string function_name ,
//		int  number_of_samples,
//		int sampling_method,
//		int problem_dimension,
//		int number_of_trials){
//
//	FILE *NNinput;
//
//	vec meanError(number_of_trials);
//	meanError.fill(0.0);
//	vec stdError(number_of_trials);
//
//
//	for(int trial=0; trial <number_of_trials; trial ++ ){
//
//		/* file name for data points in csv (comma separated values) format */
//		std::string input_file_name = function_name+"_"
//				+ std::to_string(number_of_samples )
//		+".csv";
//
//		printf("input file name : %s\n",input_file_name.c_str());
//		printf("Generating inputs using %d sample points...\n",number_of_samples );
//
//
//		/* generate the input data for test	*/
//		generate_test_function_data(test_function,
//				input_file_name,
//				bounds,
//				number_of_samples,
//				sampling_method,
//				problem_dimension);
//
//
//		/* run Neural Network Regression */
//
//		std::string python_command = "python -W ignore "+settings.python_dir+"/NeuralNetworkReg.py "+ input_file_name+ " > python.out";
//#if 1
//		printf("python_command : %s\n",python_command.c_str());
//#endif
//
//
//
//		system(python_command.c_str());
//
//
//
//		NNinput = fopen("NNoutput.dat", "r");
//
//		fscanf(NNinput,"%lf",&stdError(trial));
//		fscanf(NNinput,"%lf",&meanError(trial));
//
//		fclose(NNinput);
//
//		printf("it = %d, MSE = %10.7f, mean(MSE) = %10.7f\n",trial,meanError(trial), sum(meanError)/(trial+1));
//
//	}
//
//
//	printf("mean squared error = %10.7f\n",mean(meanError));
//
//
//}





//void perform_GEK_test(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		double *bounds,
//		std::string function_name ,
//		int  number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method,
//		int method_for_solving_lin_eq,
//		int dim,
//		int linear_regression,
//		std::string python_dir){
//
//	if(dim <= 0){
//		printf("Error: problem dimension must be greater than zero\n");
//		exit(-1);
//	}
//
//
//	vec kriging_weights;
//	vec regression_weights;
//	double reg_param;
//
//
//	int number_of_max_function_evals_for_training = 10000;
//
//	std::string input_file_name = function_name+"_"+ std::to_string(number_of_samples_with_only_f_eval)+"_"+std::to_string(number_of_samples_with_g_eval)+ ".csv";
//
//	printf("input file name : %s\n",input_file_name.c_str());
//
//
//
//	if(dim ==2){
//		/* generate the contour_plot */
//		generate_contour_plot_2D_function(test_function, bounds, function_name);
//
//	}
//
//
//	if(dim ==2){
//		printf("Generating inputs using %d points (%d gradient computations)...\n",number_of_samples_with_only_f_eval+ number_of_samples_with_g_eval,number_of_samples_with_g_eval);
//
//
//		/* generate the input data for test	*/
//		generate_2D_test_function_data_GEK(test_function,
//				test_function_adj,
//				input_file_name,
//				bounds,
//				number_of_samples_with_only_f_eval,
//				number_of_samples_with_g_eval,
//				sampling_method,
//				python_dir);
//
//	}
//
//	/* train the response surface */
//	train_GEK_response_surface(input_file_name,
//			linear_regression,
//			regression_weights,
//			kriging_weights,
//			reg_param,
//			number_of_max_function_evals_for_training,
//			dim,
//			method_for_solving_lin_eq);
//
//
//
//	printf("kriging weights = \n");
//	kriging_weights.print();
//
//
//
//
//	mat data_functional_values; // data matrix for only functional values
//	mat data_gradients;         // data matrix for only functional values + gradient sensitivities
//
//
//	std::ifstream in(input_file_name);
//
//	if(!in) {
//		cout << "Cannot open input file...\n";
//		exit(-1);
//	}
//
//
//	std::vector<double> temp;
//
//
//
//	std::string str;
//	int count_f=0;
//	int count_g=0;
//
//	while (std::getline(in, str)) {
//		// output the line
//		//				std::cout << "line = "<<str << std::endl;
//
//
//		std::string delimiter = ",";
//
//		size_t pos = 0;
//		std::string token;
//		while ((pos = str.find(delimiter)) != std::string::npos) {
//			token = str.substr(0, pos);
//			//			std::cout << "token = "<<token << std::endl;
//
//
//			temp.push_back(atof(token.c_str()));
//
//
//			str.erase(0, pos + delimiter.length());
//		}
//		//				std::cout << "str = "<<str << std::endl;
//		temp.push_back(atof(str.c_str()));
//
//		//		std::cout<<"temp= \n";
//		for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){
//
//			//			std::cout<<*it<<std::endl;
//
//
//		}
//
//
//		if(temp.size() == dim+1){ // function values
//
//			rowvec newrow(dim+1);
//			int count=0;
//			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){
//
//				//				std::cout<<*it<<std::endl;
//				newrow(count)=*it;
//				count++;
//
//			}
//			//			newrow.print();
//			count_f++;
//			data_functional_values.resize(count_f, dim+1);
//
//			data_functional_values.row(count_f-1)=newrow;
//
//
//
//		}
//		else{ // function+gradient information
//
//			rowvec newrow(2*dim+1);
//			int count=0;
//			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){
//
//				//				std::cout<<*it<<std::endl;
//				newrow(count)=*it;
//				count++;
//
//			}
//			//			newrow.print();
//			count_g++;
//			data_gradients.resize(count_g, 2*dim+1);
//
//			data_gradients.row(count_g-1)=newrow;
//
//
//		}
//
//
//
//
//		temp.clear();
//
//		// now we loop back and get the next line in 'str'
//	}
//
//
//	printf("data functional values = \n");
//	data_functional_values.print();
//	printf("data gradient sensitivities = \n");
//	data_gradients.print();
//	printf("\n");
//
//	int n_f_evals = data_functional_values.n_rows;
//	int n_g_evals = data_gradients.n_rows;
//
//
//	mat X; /* data matrix */
//	vec ys_func;
//
//	if(n_f_evals>0){
//		X = data_functional_values.submat(0, 0, n_f_evals - 1, data_functional_values.n_cols - 2);
//	}
//	//		X.print();
//
//	if(n_f_evals>0){
//		ys_func = data_functional_values.col(dim);
//	}
//
//	//	ys.print();
//
//
//	if( n_g_evals > 0){
//		X.insert_rows( n_f_evals , data_gradients.submat(0, 0, n_g_evals - 1, dim-1));
//
//		//	X.print();
//
//	}
//
//	ys_func.print();
//
//	vec x_max(dim);
//	x_max.fill(0.0);
//
//	vec x_min(dim);
//	x_min.fill(0.0);
//
//	for (int i = 0; i < dim; i++) {
//		x_max(i) = X.col(i).max();
//		x_min(i) = X.col(i).min();
//
//	}
//
//	printf("maximum = \n");
//	x_max.print();
//
//	printf("minimum = \n");
//	x_min.print();
//
//
//	/* normalize the data matrix */
//	for (unsigned int i = 0; i < X.n_rows; i++) {
//		for (int j = 0; j < dim; j++) {
//			X(i, j) = (X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//		}
//	}
//
//
//	/* normalize the data sensitivity data matrix */
//	for (unsigned int i = 0; i < data_gradients.n_rows; i++) {
//		for (int j = 0; j < dim; j++) {
//			data_gradients(i, j) = (data_gradients(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//		}
//
//		for (int j = dim+1; j < 2*dim+1; j++) {
//			data_gradients(i, j) = data_gradients(i, j) * (x_max(j-dim-1) - x_min(j-dim-1));
//		}
//	}
//
//
//	if(n_g_evals > 0){
//
//		ys_func.insert_rows(n_f_evals, data_gradients.col(dim));
//
//
//	}
//
//
//	printf("Normalized data = \n");
//	X.print();
//
//	printf("Normalized sensitivity data = \n");
//	data_gradients.print();
//
//
//
//
//
//	/* matrix that holds the gradient sensitivities */
//	mat grad(n_g_evals, dim);
//
//
//	/* copy only gradient sensitivities */
//	if( n_g_evals > 0) {
//		grad = data_gradients.submat(0, dim+1, n_g_evals - 1, 2*dim);
//	}
//
//
//	/* dimension of the correlation matrix R */
//	int dimension_of_R = n_f_evals+n_g_evals+ n_g_evals*dim;
//
//
//
//
//
//	if (linear_regression == 1) { /* if linear regression is on */
//
//		mat augmented_X(X.n_rows, dim + 1);
//
//		for (unsigned int i = 0; i < X.n_rows; i++) {
//			for (int j = 0; j <= dim; j++) {
//				if (j == 0)
//					augmented_X(i, j) = 1.0;
//				else
//					augmented_X(i, j) = X(i, j - 1);
//
//			}
//		}
//
//
//		/* now update the ys_func vector */
//
//		ys_func = ys_func - augmented_X * regression_weights;
//
//		//		printf("Updated ys vector = \n");
//
//		//		ys.print();
//
//	} /* end of linear regression */
//
//
//
//	vec ys(dimension_of_R);
//
//	/* copy n_f_evals+n_g_evals functional values from ys_func */
//	for(int i=0; i<n_f_evals+n_g_evals; i++ ) {
//		ys(i)=ys_func(i);
//	}
//
//
//	/* for each data point with gradient information*/
//	int pos = n_f_evals+n_g_evals;
//
//	for(int j=0; j<dim; j++){ /* for each design variable */
//		for(int i=0; i<n_g_evals; i++ ) {
//
//
//			ys(pos + n_g_evals*j+i) = grad(i,j);
//
//		}
//
//	}
//
//
//	printf("ys = \n");
//	ys.print();
//
//
//
//
//	/* allocate the correlation matrix */
//	mat R= eye(dimension_of_R, dimension_of_R);
//	//	R.fill(0.0);
//
//
//
//
//	/* initialize the diagonal to 1.0 plus a small regularization term*/
//	//	for (int i = 0; i < dimension_of_R; i++) {
//	//		R(i, i) = 1.0 + REGULARIZATION_TERM;
//	//	}
//
//
//
//
//	vec theta = kriging_weights.head(dim);
//
//
//	compute_R_matrix_GEK(theta, reg_param, R, X,grad);
//
//	//	R.print();
//
//	//	printf("R = \n");
//	//	R.print();
//
//
//
//	printf("condition number of R = %10.7e\n", cond(R));
//
//
//	//	mat Rinv = inv_sympd(R);
//
//
//
//
//
//
//	//		Rinv.print();
//
//
//
//
//
//
//
//	mat U(dimension_of_R, dimension_of_R);
//	mat D(dimension_of_R, dimension_of_R);
//	mat L(dimension_of_R, dimension_of_R);
//	mat V(dimension_of_R, dimension_of_R);
//	mat Ut(dimension_of_R, dimension_of_R);
//
//	vec s(dimension_of_R);
//	vec sinv(dimension_of_R);
//
//	/* allocate the F vector */
//	vec F(dimension_of_R);
//	F.fill(0.0);
//
//	/* set first n_f_evals entries to one */
//	for(int i=0; i<n_f_evals+n_g_evals ;i++) F(i)=1.0;
//
//
//
//
//	//	printf("F = \n");
//	//	F.print();
//	//	printf("\n");
//
//
//	if( method_for_solving_lin_eq == SVD){
//		int flag_svd = svd( U, s, V, R );
//
//		if (flag_svd == 0) {
//			printf("SVD could not be performed\n");
//
//			exit(-1);
//		}
//
//		sinv = 1.0/s;
//
//		sinv.print();
//
//		double threshold = 10E-08;
//		for(int i=0; i< dimension_of_R; i++){
//			if(s(i)  < threshold){
//				sinv(i) = 0.0;
//			}
//
//		}
//
//		printf("sinv = \n");
//		sinv.print();
//		printf("\n");
//
//
//		Ut = trans(U);
//
//		D.fill(0.0);
//		for(int i=0; i< dimension_of_R;i++){
//
//			D(i,i) = sinv(i);
//
//
//		}
//
//	}
//
//
//	if( method_for_solving_lin_eq == CHOLESKY){
//
//		/* compute Cholesky decomposition */
//		int flag = chol(U, R);
//
//		if (flag == 0) {
//			printf("Ill conditioned correlation matrix in Cholesky decomposition\n");
//
//			exit(-1);
//		}
//
//		L = trans(U);
//
//
//
//	}
//
//
//	vec R_inv_ys(dimension_of_R); /* R^-1* ys */
//	vec R_inv_F (dimension_of_R); /* R^-1* F */
//	vec R_inv_ys_min_beta(dimension_of_R); /* R^-1* (ys-beta0*F) */
//
//	if( method_for_solving_lin_eq == SVD){
//		R_inv_ys =V*D*Ut*ys;
//		R_inv_F  =V*D*Ut*F;
//	}
//
//	if( method_for_solving_lin_eq == CHOLESKY){
//
//		solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys);
//		solve_linear_system_by_Cholesky(U, L, R_inv_F, F);
//	}
//
//
//	/*
//		vec res1 = R*R_inv_F-F;
//
//		printf("check = R*R_inv_F-F = \n");
//		res1.print();
//		printf("\n");
//
//		vec res2 = R*R_inv_ys-ys;
//
//		printf("check = R*R_inv_ys-ys = \n");
//		res2.print();
//		printf("\n");
//
//	 */
//
//	double beta0 = (1.0/dot(F,R_inv_F)) * (dot(F,R_inv_ys));
//
//
//	vec ys_min_betaF = ys-beta0*F;
//
//	if( method_for_solving_lin_eq == CHOLESKY){
//
//		solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaF);
//
//	}
//
//	if( method_for_solving_lin_eq == SVD){
//		R_inv_ys_min_beta = V*D*Ut*ys_min_betaF;
//
//	}
//
//
//	/*
//
//		vec res3 = R*R_inv_ys_min_beta-ys_min_betaF;
//
//		printf("check R*R_inv_ys_min_beta-(ys-beta0*F) = \n");
//		res3.print();
//		printf("\n");
//
//	 */
//
//
//
//	printf("beta0 = %10.7f\n",beta0);
//
//
//
//	double in_sample_error = 0.0;
//	for(int i=0;i<X.n_rows;i++){
//
//		rowvec x = X.row(i);
//		rowvec xp = X.row(i);
//
//
//		printf("\nData point = %d\n", i+1);
//		printf("calling f_tilde at x = %10.7f y = %10.7f\n",
//				x(0)* (x_max(0) - x_min(0))+x_min(0),
//				x(1)* (x_max(1) - x_min(1))+x_min(1));
//		double func_val = calculate_f_tilde_GEK(x,
//				X,
//				beta0,
//				regression_weights,
//				R_inv_ys_min_beta,
//				theta,
//				n_g_evals );
//
//		/* perturb the first design point */
//		xp(0)+= 0.000001;
//
//		double func_valp = calculate_f_tilde_GEK(xp,
//				X,
//				beta0,
//				regression_weights,
//				R_inv_ys_min_beta,
//				theta,
//				n_g_evals );
//
//		double der_val1 = (func_valp-func_val)/0.000001;
//		xp(0)-= 0.000001;
//
//		/* perturb the second design point */
//		xp(1)+= 0.000001;
//
//		func_valp = calculate_f_tilde_GEK(xp,
//				X,
//				beta0,
//				regression_weights,
//				R_inv_ys_min_beta,
//				theta,
//				n_g_evals );
//
//		double der_val2 = (func_valp-func_val)/0.000001;
//
//
//		/* change to original coordinates before calling the original function */
//		x(0) = x(0)* (x_max(0) - x_min(0))+x_min(0);
//		x(1) = x(1)* (x_max(1) - x_min(1))+x_min(1);
//
//		double func_val_exact = test_function(x.memptr());
//
//		x(0) += 0.000001;
//
//		double func_val_exactp = test_function(x.memptr());
//
//		double der_val_f1 = (func_val_exactp-func_val_exact)/0.000001;
//		x(0) -= 0.000001;
//
//		x(1) += 0.000001;
//
//		func_val_exactp = test_function(x.memptr());
//
//		double der_val_f2 = (func_val_exactp-func_val_exact)/0.000001;
//
//
//		printf("func_val (approx) = %10.7f\n", func_val);
//		printf("func_val (exact) = %10.7f\n", func_val_exact);
//		printf("der_val1  (approx) = %10.7f\n", der_val1/(x_max(0) - x_min(0)));
//		printf("der_valf1  (approx) = %10.7f\n", der_val_f1);
//		printf("der_val2  (approx) = %10.7f\n", der_val2/(x_max(1) - x_min(1)));
//		printf("der_valf2  (approx) = %10.7f\n\n", der_val_f2);
//
//
//		in_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);
//
//		printf("in sample error = %10.7f\n", in_sample_error);
//
//	}
//
//	in_sample_error = sqrt(in_sample_error/X.n_rows);
//
//	printf("in sample error = %10.7f\n",in_sample_error);
//
//
//
//
//	if(dim ==2){
//
//		int resolution =100;
//
//
//		std::string kriging_response_surface_file_name = function_name+"_"+"kriging_response_surface_"+ std::to_string(number_of_samples_with_only_f_eval)+"_"+std::to_string(number_of_samples_with_g_eval)+".dat";
//
//		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");
//
//		double dx,dy; /* step sizes in x and y directions */
//		rowvec x(2);
//		rowvec xnorm(2);
//
//		double func_val;
//		dx = (x_max(0) - x_min(0))/(resolution-1);
//		dy = (x_max(1) - x_min(1))/(resolution-1);
//
//		dx = (80.0 - 20.0)/(resolution-1);
//		dy = (80.0 - 20.0)/(resolution-1);
//
//
//		double out_sample_error = 0.0;
//
//		x[0] = x_min(0);
//		x[0] = 20.0;
//		for(int i=0;i<resolution;i++){
//			x[1] = x_min(1);
//			x[1] = 20.0;
//			for(int j=0;j<resolution;j++){
//
//				/* normalize x */
//				xnorm(0)= (x(0)- x_min(0)) / (x_max(0) - x_min(0));
//				xnorm(1)= (x(1)- x_min(1)) / (x_max(1) - x_min(1));
//
//				//				printf("calling f_tilde at x = %10.7f y = %10.7f\n", x(0),x(1));
//				func_val = calculate_f_tilde_GEK(xnorm,
//						X,
//						beta0,
//						regression_weights,
//						R_inv_ys_min_beta,
//						kriging_weights,
//						n_g_evals );
//
//				double func_val_exact = test_function(x.memptr());
//
//				out_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);
//				//				printf("out of sample error = %10.7f\n",out_sample_error);
//				//				printf("x = %10.7f y = %10.7f ftilde = %10.7f fexact = %10.7f\n",x(0),x(1),func_val,func_val_exact);
//				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),func_val);
//
//				x[1]+=dy;
//			}
//			x[0]+= dx;
//
//		}
//		fclose(kriging_response_surface_file);
//
//		out_sample_error = sqrt(out_sample_error/(resolution*resolution));
//
//		printf("out of sample error = %10.7f\n",out_sample_error);
//
//
//		/* plot the kriging response surface */
//
//		std::string file_name_for_plot = function_name+"_"+"kriging_response_surface_";
//		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";
//
//		std::string python_command = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_file_name+ " "+ file_name_for_plot ;
//
//
//
//		FILE *figure_out = popen(python_command.c_str(), "r");
//
//
//		fprintf(figure_out, "\n");
//
//
//	}
//
//
//}









//void generate_2D_test_function_data(double (*test_function)(double *),
//		std::string filename,
//		double bounds[4],
//		int number_of_function_evals,
//		int sampling_method){
//
//	if(sampling_method == EXISTING_FILE){
//		/* do nothing */
//
//
//	}
//
//
//	if(sampling_method == RANDOM_SAMPLING){
//
//		FILE *outp;
//
//		//	printf("opening file %s for input...\n",filename.c_str() );
//		outp = fopen(filename.c_str(), "w");
//
//
//
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//		double ys = bounds[2];
//		double ye = bounds[3];
//
//		double x[2];
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			x[0] = RandomDouble(xs, xe);
//			x[1] = RandomDouble(ys, ye);
//			double f_val = test_function(x);
//
//			fprintf(outp,"%10.7f %10.7f %10.7f\n",x[0],x[1],f_val);
//
//		}
//
//		fclose(outp);
//	}
//
//
//	if(sampling_method == LHS_CENTER){
//
//		double x[2];
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//		double ys = bounds[2];
//		double ye = bounds[3];
//
//		std::string lhs_filename = "lhs_points.dat";
//		std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ "2" + " "+ std::to_string(number_of_function_evals)+ " center" ;
//
//		system(python_command.c_str());
//
//		FILE *inp= fopen("lhs_points.dat","r");
//		FILE *outp = fopen(filename.c_str(), "w");
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			fscanf(inp,"%lf %lf",&x[0],&x[1]);
//			x[0] = x[0]*(xe-xs)+xs;
//			x[1] = x[1]*(ye-ys)+ys;
//			double f_val = test_function(x);
//
//			fprintf(outp,"%10.7f %10.7f %10.7f\n",x[0],x[1],f_val);
//
//		}
//
//		fclose(inp);
//		fclose(outp);
//
//
//	}
//
//	if(sampling_method == LHS_RANDOM){
//
//		double x[2];
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//		double ys = bounds[2];
//		double ye = bounds[3];
//
//		std::string lhs_filename = "lhs_points.dat";
//		std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ "2" + " "+ std::to_string(number_of_function_evals)+ " None" ;
//
//		system(python_command.c_str());
//
//		FILE *inp= fopen("lhs_points.dat","r");
//		FILE *outp = fopen(filename.c_str(), "w");
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			fscanf(inp,"%lf %lf",&x[0],&x[1]);
//			x[0] = x[0]*(xe-xs)+xs;
//			x[1] = x[1]*(ye-ys)+ys;
//			double f_val = test_function(x);
//
//			fprintf(outp,"%10.7f %10.7f %10.7f\n",x[0],x[1],f_val);
//
//		}
//
//		fclose(inp);
//		fclose(outp);
//
//
//	}
//
//}


//void generate_test_function_data(double (*test_function)(double *),
//		std::string filename,
//		double * bounds,
//		int number_of_function_evals,
//		int sampling_method,
//		int problem_dimension){
//
//
//	double *xs = new double[problem_dimension];
//	double *xe = new double[problem_dimension];
//	double *x  = new double[problem_dimension];
//
//	int count=0;
//	for(int i=0;i<problem_dimension;i++){
//		xs[i] = bounds[count];
//		count++;
//		xe[i] = bounds[count];
//		count++;
//
//	}
//
//
//	if(sampling_method == RANDOM_SAMPLING){
//
//		FILE *outp;
//#if 0
//		printf("opening file %s for input...\n",filename.c_str() );
//#endif
//		outp = fopen(filename.c_str(), "w");
//
//
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			for(int j=0;j<problem_dimension;j++) x[j] = RandomDouble(xs[j], xe[j]);
//
//			double f_val = test_function(x);
//
//			for(int j=0;j<problem_dimension;j++) fprintf(outp,"%10.7f, ",x[j]);
//
//			fprintf(outp,"%10.7f\n",f_val);
//
//		}
//
//		fclose(outp);
//	}
//
//
//	if(sampling_method == LHS_CENTER){
//
//
//		std::string str_problem_dim = std::to_string(problem_dimension);
//		std::string lhs_filename = "lhs_points.dat";
//		std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(number_of_function_evals)+ " center" ;
//
//		system(python_command.c_str());
//
//		FILE *inp= fopen("lhs_points.dat","r");
//		FILE *outp = fopen(filename.c_str(), "w");
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			for(int j=0;j<problem_dimension;j++) fscanf(inp,"%lf",&x[j]);
//
//			for(int j=0;j<problem_dimension;j++) x[j] = x[j]*(xe[j]-xs[j])+xs[j];
//
//
//			double f_val = test_function(x);
//
//			for(int j=0;j<problem_dimension;j++) fprintf(outp,"%10.7f, ",x[j]);
//
//			fprintf(outp,"%10.7f\n",f_val);
//
//		}
//
//		fclose(inp);
//		fclose(outp);
//
//
//
//
//	}
//
//	if(sampling_method == LHS_RANDOM){
//
//
//		std::string str_problem_dim = std::to_string(problem_dimension);
//		std::string lhs_filename = "lhs_points.dat";
//		std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(number_of_function_evals)+ " None" ;
//
//		system(python_command.c_str());
//
//		FILE *inp= fopen("lhs_points.dat","r");
//		FILE *outp = fopen(filename.c_str(), "w");
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			for(int j=0;j<problem_dimension;j++) fscanf(inp,"%lf",&x[j]);
//
//			for(int j=0;j<problem_dimension;j++) x[j] = x[j]*(xe[j]-xs[j])+xs[j];
//
//
//			double f_val = test_function(x);
//
//			for(int j=0;j<problem_dimension;j++) fprintf(outp,"%10.7f, ",x[i]);
//
//			fprintf(outp,"%10.7f\n",f_val);
//
//		}
//
//		fclose(inp);
//		fclose(outp);
//
//
//
//
//	}
//
//	if(sampling_method == EXISTING_FILE){
//		/* do nothing */
//
//
//	}
//
//}




//void generate_highdim_test_function_data_GEK(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		std::string filename,
//		double *bounds,
//		int dim,
//		int number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method){
//
//#if 0
//	printf("generate_highdim_test_function_data_GEK...\n");
//	printf("dim = %d\n",dim);
//#endif
//
//	int number_of_function_evals  =  number_of_samples_with_only_f_eval+ number_of_samples_with_g_eval;
//
//
//	if(sampling_method == EXISTING_FILE){
//
//		/* do nothing */
//
//	}
//
//	if(sampling_method == RANDOM_SAMPLING){
//
//		FILE *outp;
//#if 0
//		printf("opening file %s for input...\n",filename.c_str() );
//#endif
//		outp = fopen(filename.c_str(), "w");
//
//		double *x = new double[dim];
//		double *xb = new double[dim];
//
//		/* write functional values to the output file */
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			for(int j=0; j<dim;j++){
//#if 0
//				printf("bounds[%d] = %10.7f\n",j*2,bounds[j*2]);
//				printf("bounds[%d] = %10.7f\n",j*2+1,bounds[j*2+1]);
//#endif
//				x[j] = RandomDouble(bounds[j*2], bounds[j*2+1]);
//			}
//
//			double f_val = test_function(x);
//
//			for(int j=0; j<dim;j++){
//				printf("%10.7f, ",x[j]);
//				fprintf(outp,"%10.7f, ",x[j]);
//			}
//			printf("%10.7f\n",f_val);
//			fprintf(outp,"%10.7f\n",f_val);
//
//
//		}
//
//
//		/* write functional values and the gradient sensitivities to the output file */
//
//		for(int i=number_of_samples_with_only_f_eval; i<number_of_function_evals;i++ ){
//
//			for(int j=0; j<dim;j++){
//#if 0
//				printf("bounds[%d] = %10.7f\n",j*2,bounds[j*2]);
//				printf("bounds[%d] = %10.7f\n",j*2+1,bounds[j*2+1]);
//#endif
//
//				x[j] = RandomDouble(bounds[j*2], bounds[j*2+1]);
//				xb[j]=0.0;
//			}
//
//			double f_val = test_function_adj(x,xb);
//
//#if 0
//			double f_val_original = test_function(x);
//
//			printf("fval = %10.7f\n",f_val);
//			printf("fval(original) = %10.7f\n",f_val_original);
//
//			for(int j=0; j<dim;j++){
//				double eps = x[j]*0.0001;
//				x[j]+= eps;
//				double fplus = test_function(x);
//				x[j]-= eps;
//
//				double fdval = (fplus-f_val_original)/eps;
//				printf("fd = %10.7f\n",fdval);
//				printf("adj = %10.7f\n",xb[j]);
//
//
//			}
//#endif
//
//
//
//			for(int j=0; j<dim;j++){
//#if 0
//				printf("%10.7f, ",x[j]);
//#endif
//				fprintf(outp,"%10.7f, ",x[j]);
//			}
//#if 0
//			printf("%10.7f, ",f_val);
//#endif
//			fprintf(outp,"%10.7f, ",f_val);
//
//			for(int j=0; j<dim-1;j++){
//#if 0
//				printf("%10.7f, ",xb[j]);
//#endif
//				fprintf(outp,"%10.7f, ",xb[j]);
//			}
//#if 0
//			printf("%10.7f\n",xb[dim-1]);
//#endif
//			fprintf(outp,"%10.7f\n",xb[dim-1]);
//
//
//		}
//
//		fclose(outp);
//
//		delete[] x;
//		delete[] xb;
//
//	}
//
//
//}
//
//
//void generate_highdim_test_function_data_cuda(double (*test_function)(double *),
//		std::string filename,
//		double *bounds,
//		int number_of_samples,
//		int dim){
//
//#if 0
//	printf("generate_highdim_test_function_data...\n");
//	printf("dim = %d\n",dim);
//#endif
//
//	FILE *outp;
//#if 0
//	printf("opening file %s for input...\n",filename.c_str() );
//#endif
//	outp = fopen(filename.c_str(), "w");
//
//	double *x = new double[dim];
//
//
//	/* write functional values to the output file */
//	for(int i=0; i<number_of_samples;i++ ){
//
//		for(int j=0; j<dim;j++){
//#if 0
//			printf("bounds[%d] = %10.7f\n",j*2,bounds[j*2]);
//			printf("bounds[%d] = %10.7f\n",j*2+1,bounds[j*2+1]);
//#endif
//			x[j] = RandomDouble(bounds[j*2], bounds[j*2+1]);
//		}
//
//		double f_val = test_function(x);
//
//		for(int j=0; j<dim;j++){
//#if 0
//			printf("%10.7f, ",x[j]);
//#endif
//			fprintf(outp,"%10.7f, ",x[j]);
//		}
//#if 0
//		printf("%10.7f\n",f_val);
//#endif
//		fprintf(outp,"%10.7f\n",f_val);
//
//
//	}
//
//	fclose(outp);
//
//	delete[] x;
//
//}
//
//
//
//
//void generate_2D_test_function_data_GEK(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		std::string filename,
//		double bounds[4],
//		int number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method,
//		std::string python_dir){
//
//
//	int number_of_function_evals  =  number_of_samples_with_only_f_eval
//			+ number_of_samples_with_g_eval;
//
//
//	if(sampling_method == EXISTING_FILE){
//
//		/* do nothing */
//
//	}
//
//
//	if(sampling_method == LHS_CENTER){
//		printf("Generating LHS_CENTER samples ...\n");
//
//		vec x(number_of_function_evals);
//		vec y(number_of_function_evals);
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//		double ys = bounds[2];
//		double ye = bounds[3];
//
//		std::string lhs_filename = "lhs_points.dat";
//		std::string python_command = "python -W ignore "+ python_dir+"/lhs.py "
//				+ lhs_filename+ " "+ "2" + " "
//				+ std::to_string(number_of_function_evals)
//		+ " center" ;
//
//#if 0
//		printf("python_command = %s\n",python_command.c_str());
//#endif
//
//		system(python_command.c_str());
//
//		FILE *inp= fopen("lhs_points.dat","r");
//		FILE *outp = fopen(filename.c_str(), "w");
//
//
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			fscanf(inp,"%lf %lf\n",&x(i),&y(i));
//
//
//		}
//		fclose(inp);
//
//
//		/* write functional values to the output file */
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			double xin[2];
//			xin[0]=x[i]*(xe-xs)+xs;
//			xin[1]=y[i]*(ye-ys)+ys;
//			double f_val = test_function(xin);
//#if 0
//			printf("%10.7f, %10.7f, %10.7f \n",x[i],y[i],f_val);
//#endif
//			fprintf(outp,"%10.7f, %10.7f, %10.7f \n",x[i],y[i],f_val);
//
//		}
//
//		/* write functional values and the gradient sensitivities to the output file */
//
//		for(int i=number_of_samples_with_only_f_eval; i<number_of_function_evals;i++ ){
//
//			double xin[2];
//			double xb[2];
//			xb[0]=0.0;
//			xb[1]=0.0;
//			xin[0]=x[i]*(xe-xs)+xs;
//			xin[1]=y[i]*(ye-ys)+ys;
//			double f_val = test_function_adj(xin,xb);
//
//#if 0
//			printf("%10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",xin[0],xin[1],f_val,xb[0],xb[1]);
//#endif
//			fprintf(outp,"%10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",xin[0],xin[1],f_val,xb[0],xb[1]);
//
//		}
//
//
//
//
//		fclose(outp);
//
//
//
//
//	}
//
//
//
//
//
//	if(sampling_method == RANDOM_SAMPLING){
//#if 0
//		printf("Generating random samples ...\n");
//#endif
//		FILE *outp;
//#if 0
//		printf("opening file %s for input...\n",filename.c_str() );
//#endif
//		outp = fopen(filename.c_str(), "w");
//
//
//
//		/* parameter bounds */
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//		double ys = bounds[2];
//		double ye = bounds[3];
//
//
//		vec x(number_of_function_evals);
//		vec y(number_of_function_evals);
//
//		/* generate random points */
//		for(int i=0; i<number_of_function_evals;i++ ){
//
//			x[i] = RandomDouble(xs, xe);
//			y[i] = RandomDouble(ys, ye);
//
//
//		}
//
//
//
//
//		/* write functional values to the output file */
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			double xin[2];
//			xin[0]=x[i];
//			xin[1]=y[i];
//			double f_val = test_function(xin);
//#if 0
//			printf("%10.7f, %10.7f, %10.7f \n",x[i],y[i],f_val);
//#endif
//			fprintf(outp,"%10.7f, %10.7f, %10.7f \n",x[i],y[i],f_val);
//
//		}
//
//
//		/* write functional values and the gradient sensitivities to the output file */
//
//		for(int i=number_of_samples_with_only_f_eval; i<number_of_function_evals;i++ ){
//
//			double xin[2];
//			double xb[2];
//			xb[0]=0.0;
//			xb[1]=0.0;
//			xin[0]=x[i];
//			xin[1]=y[i];
//			double f_val = test_function_adj(xin,xb);
//
//#if 0
//			printf("%10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",x[i],y[i],f_val,xb[0],xb[1]);
//#endif
//			fprintf(outp,"%10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",x[i],y[i],f_val,xb[0],xb[1]);
//
//		}
//
//
//
//		fclose(outp);
//
//
//
//	} /* end of random sampling */
//
//
//
//
//
//}
//
//
//void generate_1D_test_function_data_GEK(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		std::string filename,
//		double *bounds,
//		int number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method,
//		double *locations_func,
//		double *locations_grad){
//
//
//	int number_of_function_evals  =  number_of_samples_with_only_f_eval+ number_of_samples_with_g_eval;
//
//
//	if(sampling_method == PREDEFINED_LOCATIONS){
//
//
//		double max = -LARGE;
//		double min =  LARGE;
//
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			if(locations_func[i] > max) max = locations_func[i];
//			if(locations_func[i] < min) min = locations_func[i];
//		}
//
//
//
//		for(int i=0; i<number_of_samples_with_g_eval;i++ ){
//
//			if(locations_grad[i] > max) max = locations_grad[i];
//			if(locations_grad[i] < min) min = locations_grad[i];
//		}
//
//
//
//
//		FILE *outp;
//
//		printf("opening file %s for input...\n",filename.c_str() );
//		outp = fopen(filename.c_str(), "w");
//
//
//
//		double x;
//
//		/* write functional values to the output file */
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			x = locations_func[i];
//
//			double f_val = test_function(&x);
//
//			fprintf(outp,"%10.7f, %10.7f\n",x,f_val);
//			printf("%d %10.7f %10.7f\n",i,x,f_val);
//
//		}
//
//		double xb;
//
//		/* write functional values and the gradient sensitivites to the output file */
//		for(int i=0; i<number_of_samples_with_g_eval;i++ ){
//
//			xb=0.0;
//
//			x = locations_grad[i];
//			double f_val = test_function_adj(&x,&xb);
//
//
//
//
//			fprintf(outp,"%10.7f, %10.7f, %10.7f\n",x,f_val,xb);
//			printf("%d %10.7f, %10.7f, %10.7f\n",i,x,f_val,xb);
//
//		}
//
//
//
//		fclose(outp);
//
//
//	}
//
//
//
//	if(sampling_method == RANDOM_SAMPLING){
//
//		FILE *outp;
//
//		printf("opening file %s for input...\n",filename.c_str() );
//		outp = fopen(filename.c_str(), "w");
//
//
//
//		double xs = bounds[0];
//		double xe = bounds[1];
//
//
//		vec x(number_of_samples_with_only_f_eval+number_of_samples_with_g_eval);
//
//		for(int i=0; i<number_of_samples_with_only_f_eval+number_of_samples_with_g_eval;i++ ){
//
//			x(i) = RandomDouble(xs, xe);
//
//
//		}
//
//		/* write functional values to the output file */
//		for(int i=0; i<number_of_samples_with_only_f_eval;i++ ){
//
//			double f_val = test_function(&x(i));
//
//			fprintf(outp,"%10.7f, %10.7f\n",x(i),f_val);
//
//		}
//
//		double xb;
//
//		/* write functional values and the gradient sensitivites to the output file */
//		for(int i=number_of_samples_with_only_f_eval; i<number_of_samples_with_g_eval+number_of_samples_with_only_f_eval;i++ ){
//
//			xb=0.0;
//
//			double f_val = test_function_adj(&x(i),&xb);
//
//			fprintf(outp,"%10.7f, %10.7f, %10.7f\n",x(i),f_val,xb);
//
//		}
//
//		fclose(outp);
//
//
//	} /* end of random sampling */
//
//
//}


//void perform_trust_region_GEK_test(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		double *bounds,
//		std::string function_name ,
//		int  number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method,
//		int method_for_solving_lin_eq,
//		int dim,
//		int linear_regression,
//		std::string python_dir){
//
//	if(dim <= 0){
//		printf("Error: problem dimension must be greater than zero\n");
//		exit(-1);
//	}
//
//
//	/* Kriging specific parameters */
//
//	mat kriging_weights;
//	mat regression_weights;
//	mat R_inv_ys_min_beta;
//
//	vec beta0(dim+1);
//
//
//	int number_of_max_function_evals_for_training = 10000;
//
//	std::string input_file_name = function_name+"_"+ std::to_string(number_of_samples_with_only_f_eval)+"_"+std::to_string(number_of_samples_with_g_eval)+ ".csv";
//
//	printf("input file name : %s\n",input_file_name.c_str());
//
//
//
//	if(dim ==2){
//		/* generate the contour_plot */
//		generate_contour_plot_2D_function_with_gradient(test_function_adj, bounds, function_name, python_dir);
//
//
//	}
//
//
//	if(dim == 2){
//		printf("Generating inputs using %d points (%d gradient computations)...\n",number_of_samples_with_only_f_eval+ number_of_samples_with_g_eval,number_of_samples_with_g_eval);
//
//
//		/* generate the input data for test	*/
//		generate_2D_test_function_data_GEK(test_function,
//				test_function_adj,
//				input_file_name,
//				bounds,
//				number_of_samples_with_only_f_eval,
//				number_of_samples_with_g_eval,
//				sampling_method,
//				python_dir);
//
//	}
//	else{
//
//		/* generate the input data for test	*/
//		generate_highdim_test_function_data_GEK(test_function,
//				test_function_adj,
//				input_file_name,
//				bounds,
//				dim,
//				number_of_samples_with_only_f_eval,
//				number_of_samples_with_g_eval,
//				sampling_method );
//
//	}
//
//
//
//
//	double radius;
//
//#if 0
//	printf("Training the hyperparameters of the model...\n");
//#endif
//	/* train the response surfaces */
//	train_TRGEK_response_surface(input_file_name,
//			"None",
//			linear_regression,
//			regression_weights,
//			kriging_weights,
//			R_inv_ys_min_beta,
//			radius,
//			beta0,
//			number_of_max_function_evals_for_training,
//			dim,1);
//
//
//}

//void perform_kernel_regression_test_highdim(double (*test_function)(double *),
//		double *bounds,
//		std::string function_name,
//		int number_of_samples,
//		int sampling_method,
//		int dim){
//
//	double sigma = 0.01;
//	double wSvd = 1.0;
//	double w12 = 1.0;
//	int max_cv_iter = 50;
//	int number_of_trials = 1;
//
//	vec genErr1(number_of_trials); genErr1.fill(0.0);
//	vec genErr2(number_of_trials); genErr2.fill(0.0);
//
//	std::string datafilename;
//
//	if (sampling_method == EXISTING_FILE){
//
//		datafilename = function_name + ".csv";
//
//
//	}
//	else{
//
//		datafilename = function_name+"_"+ std::to_string(number_of_samples)+".csv";
//
//
//	}
//
//#if 1
//	printf("filename = %s\n",datafilename.c_str());
//
//#endif
//	/* division between number of training and validation samples */
//	double partition[2] = {0.1,0.9};
//
//	srand (time(NULL));
//
//
//	for(int iter=0; iter<number_of_trials; iter++){
//
//#if 0
//		printf("iter = %d\n",iter);
//#endif
//
//		if ( sampling_method == EXISTING_FILE ){
//
//			// do nothing
//
//		}
//		else{
//
//			generate_highdim_test_function_data_cuda(test_function,
//					datafilename,
//					bounds,
//					number_of_samples,
//					dim);
//
//		} /* end of else */
//
//
//
//		mat data;
//		bool load_ok = data.load(datafilename);
//
//		if(load_ok == false)
//		{
//			printf("problem with loading the file %s\n",datafilename.c_str());
//			exit(-1);
//		}
//
//
//#if 1
//		printf("Kernel regression with the input data: %s\n",datafilename.c_str());
//		printf("Data has %d samples with %d variables\n",number_of_samples,dim);
//		printf("Original data:\n");
//		data.print();
//#endif
//
//
//		int number_of_training_samples = number_of_samples*partition[0];
//		int number_of_test_samples = number_of_samples - number_of_training_samples;
//
//#if 0
//		printf("number of training samples = %d\n",number_of_training_samples);
//		printf("number of test samples = %d\n",number_of_test_samples);
//#endif
//
//		mat dataTraining;
//		mat dataTest;
//
//		dataTraining = data.submat( 0, 0, number_of_training_samples-1, dim );
//
//		dataTest     = data.submat( number_of_training_samples, 0, number_of_samples-1, dim );
//
//
//		data.reset();
//
//#if 1
//		printf("Training data:\n");
//		dataTraining.print();
//
//		printf("Test data:\n");
//		dataTest.print();
//#endif
//
//
//		vec x_maxTraining(dim);
//		x_maxTraining.fill(0.0);
//
//		vec x_minTraining(dim);
//		x_minTraining.fill(0.0);
//
//		for (int i = 0; i < dim; i++) {
//
//			x_maxTraining(i) = dataTraining.col(i).max();
//			x_minTraining(i) = dataTraining.col(i).min();
//
//		}
//
//#if 0
//		printf("maximum = \n");
//		x_maxTraining.print();
//
//		printf("minimum = \n");
//		x_minTraining.print();
//#endif
//
//		/* normalize functional values */
//
//		double yTrainingMax = 1.0;
//
//		yTrainingMax = dataTraining.col(dim).max();
//
//		for (int i = 0; i < number_of_training_samples; i++) {
//
//			dataTraining(i, dim) = dataTraining(i, dim)/yTrainingMax ;
//
//		}
//
//
//		/* normalize training data */
//		for (int i = 0; i < number_of_training_samples; i++) {
//
//			for (int j = 0; j < dim; j++) {
//
//				dataTraining(i, j) = (dataTraining(i, j) - x_minTraining(j)) / (x_maxTraining(j) - x_minTraining(j));
//			}
//
//		}
//
//
//
//#if 1
//		printf("Training data (normalized) = \n");
//		dataTraining.print();
//#endif
//
//		vec x_maxTest(dim); x_maxTest.fill(0.0);
//		vec x_minTest(dim); x_minTest.fill(0.0);
//
//		for (int i = 0; i < dim; i++) {
//
//			x_maxTest(i) = dataTest.col(i).max();
//			x_minTest(i) = dataTest.col(i).min();
//
//		}
//
//#if 0
//		printf("maximum = \n");
//		x_maxTest.print();
//
//		printf("minimum = \n");
//		x_minTest.print();
//#endif
//
//
//		/* normalize test data */
//		for (int i = 0; i < number_of_test_samples; i++) {
//
//			for (int j = 0; j < dim; j++) {
//
//				dataTest(i, j) = (dataTest(i, j) - x_minTest(j)) / (x_maxTest(j) - x_minTest(j));
//			}
//
//		}
//
//#if 1
//		printf("Test data (normalized) = \n");
//		dataTest.print();
//#endif
//
//		mat L(dim,dim);
//		L.fill(0.0);
//
//
//	trainMahalanobisDistance(L, dataTraining, sigma, wSvd, w12, max_cv_iter,L1_LOSS_FUNCTION, number_of_training_samples, 50000);
//
//#if 1
//
//		printf("L = \n");
//		L.print();
//		printf("sigma = %10.7f\n",sigma);
//#endif
//
//
//		/* from optimal L we obtain now M */
//
//		mat M = L*trans(L);
//
//#if 0
//		/* give singular value info */
//
//		mat U;
//		vec s;
//		mat V;
//
//		svd(U,s,V,M);
//
//
//
//		printf("M = \n");
//		M.print();
//		printf("sigma = %10.7f\n",sigma);
//		printf("singular values of M = \n");
//		s.print();
//#endif
//
//
//
//		/* compute generalization error */
//
//		mat XTest = dataTest.submat(0,0,number_of_test_samples-1,dim-1);
//		vec yTest = dataTest.col(dim);
//
//
//#if 0
//		printf("number of test samples =%d\n",number_of_test_samples);
//		printf("XTest = \n");
//		XTest.print();
//		printf("yTest = \n");
//		trans(yTest).print();
//#endif
//		mat XTraining = dataTraining.submat(0,0,number_of_training_samples-1,dim-1);
//		vec yTraining = dataTraining.col(dim);
//
//#if 0
//		printf("dataTraining =\n");
//		dataTraining.print();
//		printf("yTraining =\n");
//		yTraining.print();
//#endif
//
//
//		double genError = 0.0;
//
//
//		for(int i=0;i <number_of_test_samples; i++){
//
//			rowvec xp = XTest.row(i);
//#if 0
//			printf("xp = \n");
//			xp.print();
//			printf("ytest (normalized) = %10.7f\n",yTest(i)/yTrainingMax);
//
//#endif
//
//			double ytilde = 0.0;
//
//			ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax ;
//
//
//			double yexact = yTest(i);
//
//
//			genError += (yexact-ytilde)*(yexact-ytilde);
//#if 0
//
//			printf("yexact = %10.7f, ytilde = %10.7f\n", yexact,ytilde);
//#endif
//
//
//
//		}
//
//
//		genError = genError/number_of_test_samples;
//		genErr1(iter) = genError;
//
//		printf("genError (out of sample)= %10.7f\n",genError );
//
//
//	}
//
//
//}
//
//
//
//void perform_kernel_regression_test_highdim(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		double *bounds,
//		std::string function_name,
//		int number_of_samples_with_only_f_eval,
//		int number_of_samples_with_g_eval,
//		int sampling_method,
//		int dim){
//
//
//	float sigma = 0.01;
//	float wSvd = 1.0;
//	float w12 = 1.0;
//	int max_cv_iter = 50;
//	int number_of_trials = 1;
//
//
//	std::string datafilename;
//
//	if (sampling_method == EXISTING_FILE){
//
//		datafilename = function_name + ".csv";
//
//
//	}
//	else{
//
//		datafilename = function_name+"_"+ std::to_string(number_of_samples_with_only_f_eval)+"_"
//				+ std::to_string(number_of_samples_with_g_eval)+ ".csv";
//
//
//	}
//
//
//
//
//	/* division between number of training and validation samples */
//	float partition[2] = {0.1,0.9};
//
//	srand (time(NULL));
//
//
//
//
//	vec genErr1(number_of_trials); genErr1.fill(0.0);
//	vec genErr2(number_of_trials); genErr2.fill(0.0);
//	vec genErr3(number_of_trials); genErr3.fill(0.0);
//	vec genErr4(number_of_trials); genErr4.fill(0.0);
//
//	for(int iter=0; iter<number_of_trials; iter++){
//
//#if 0
//		printf("iter = %d\n",iter);
//#endif
//
//		if ( sampling_method == EXISTING_FILE ){
//
//			// do nothing
//
//		}
//		else{
//
//
//			/* generate data only with functional values */
//
//			if(number_of_samples_with_g_eval == 0) {
//
//				int number_of_samples_to_generate = number_of_samples_with_only_f_eval;
//#if 1
//				printf("Generating data with %d samples...\n", number_of_samples_to_generate);
//#endif
//				generate_highdim_test_function_data_cuda(test_function,
//						datafilename,
//						bounds,
//						number_of_samples_to_generate,
//						dim);
//
//			}
//
//			else{
//
//				/* generate data with gradients */
//
//				generate_highdim_test_function_data_GEK(test_function,
//						test_function_adj,
//						datafilename,
//						bounds,
//						dim,
//						number_of_samples_with_only_f_eval,
//						number_of_samples_with_g_eval,
//						sampling_method);
//
//
//
//			}
//
//
//
//
//
//		} /* end of else */
//
//
//
//		fmat data;
//		bool load_ok = data.load(datafilename);
//
//		if(load_ok == false)
//		{
//			printf("problem with loading the file %s\n",datafilename.c_str());
//			exit(-1);
//		}
//
//
//		int number_of_data_points = data.n_rows;
//
//
//		if(number_of_samples_with_g_eval ==0  && dim != data.n_cols-1){
//
//			printf("Number of columns in the input file does not match with dim = %d\n",dim);
//			exit(-1);
//
//		}
//
//		if(number_of_samples_with_g_eval >0  && 2*dim != data.n_cols-1){
//
//			printf("Number of columns in the input file does not match with dim = %d\n",dim);
//			exit(-1);
//
//		}
//
//#if 0
//		printf("Kernel regression with the input data: %s\n",datafilename.c_str());
//		printf("Data has %d samples with %d variables\n",number_of_data_points,dim);
//#endif
//#if 0
//		printf("Original data:\n");
//		data.print();
//#endif
//
//
//
//
//		data = shuffle(data);
//#if 0
//		printf("Original data (shuffled):\n");
//		data.print();
//#endif
//
//		int number_of_training_samples = number_of_data_points*partition[0];
//		int number_of_test_samples = number_of_data_points - number_of_training_samples;
//
//#if 0
//		printf("number of training samples = %d\n",number_of_training_samples);
//		printf("number of test samples = %d\n",number_of_test_samples);
//#endif
//
//		fmat dataTraining;
//		fmat dataTest;
//
//		if(number_of_samples_with_g_eval ==0 ) {
//
//			dataTraining = data.submat( 0, 0, number_of_training_samples-1, dim );
//
//
//		}
//
//		if(number_of_samples_with_g_eval >0 ) {
//
//			dataTraining = data.submat( 0, 0, number_of_training_samples-1, 2*dim );
//
//
//		}
//
//
//		dataTest     = data.submat( number_of_training_samples, 0, number_of_data_points-1, dim );
//
//
//
//
//		fmat XTrainingNotNormalized = dataTraining.submat(0,0,number_of_training_samples-1, dim-1);
//
//
//
//		fmat gradTrainingNotNormalized;
//		if(number_of_samples_with_g_eval >0 ) {
//
//			gradTrainingNotNormalized = dataTraining.submat(0,dim+1,number_of_training_samples-1,2*dim);
//
//
//		}
//		fvec yTrainingNotNormalized = dataTraining.col(dim);
//
//		fmat XTestNotNormalized = dataTest.submat(0,0,number_of_test_samples-1,dim-1);
//
//
//
//
//
//		data.reset();
//
//#if 0
//		printf("Training data:\n");
//		dataTraining.print();
//
//		printf("Test data:\n");
//		dataTest.print();
//#endif
//
//
//
//
//
//		fvec x_maxTraining(dim);
//		x_maxTraining.fill(0.0);
//
//		fvec x_minTraining(dim);
//		x_minTraining.fill(0.0);
//
//		for (int i = 0; i < dim; i++) {
//
//			x_maxTraining(i) = dataTraining.col(i).max();
//			x_minTraining(i) = dataTraining.col(i).min();
//
//		}
//
//#if 0
//		printf("maximum = \n");
//		x_maxTraining.print();
//
//		printf("minimum = \n");
//		x_minTraining.print();
//#endif
//
//		/* normalize functional values */
//
//		float yTrainingMax = 1.0;
//
//		yTrainingMax = dataTraining.col(dim).max();
//
//		for (int i = 0; i < number_of_training_samples; i++) {
//
//			dataTraining(i, dim) = dataTraining(i, dim)/yTrainingMax ;
//
//		}
//
//
//		/* normalize training data */
//		for (int i = 0; i < number_of_training_samples; i++) {
//
//			for (int j = 0; j < dim; j++) {
//
//				dataTraining(i, j) = (1.0/dim)*(dataTraining(i, j) - x_minTraining(j)) / (x_maxTraining(j) - x_minTraining(j));
//			}
//
//
//			/* this part is for derivatives */
//			if(number_of_samples_with_g_eval >0) {
//
//
//				for (int j = dim+1; j < 2*dim+1; j++) {
//
//					dataTraining(i, j) = dataTraining(i, j) * (dim/yTrainingMax) * (x_maxTraining(j-dim-1) - x_minTraining(j-dim-1));
//				}
//
//
//
//			}
//
//		}
//
//
//
//#if 0
//		printf("Training data (normalized) = \n");
//		dataTraining.print();
//#endif
//
//
//
//
//		fvec x_maxTest(dim);
//		x_maxTest.fill(0.0);
//
//		fvec x_minTest(dim);
//		x_minTest.fill(0.0);
//
//		for (int i = 0; i < dim; i++) {
//
//			x_maxTest(i) = dataTest.col(i).max();
//			x_minTest(i) = dataTest.col(i).min();
//
//		}
//
//#if 0
//		printf("maximum = \n");
//		x_maxTest.print();
//
//		printf("minimum = \n");
//		x_minTest.print();
//#endif
//
//
//		/* normalize test data */
//		for (int i = 0; i < number_of_test_samples; i++) {
//
//			for (int j = 0; j < dim; j++) {
//
//				dataTest(i, j) = (1.0/dim)*(dataTest(i, j) - x_minTest(j)) / (x_maxTest(j) - x_minTest(j));
//			}
//
//
//
//		}
//
//#if 0
//		printf("Test data (normalized) = \n");
//		dataTest.print();
//#endif
//
//
//
//		fmat L(dim,dim);
//		L.fill(0.0);
//
//
//		//		trainMahalanobisDistance(L, dataTraining, sigma, wSvd, w12, max_cv_iter,L2_LOSS_FUNCTION, number_of_training_samples, 50000);
//
//
//
//#if 1
//
//		printf("L = \n");
//		L.print();
//		printf("sigma = %10.7f\n",sigma);
//#endif
//
//
//		/* from optimal L we obtain now M */
//
//		fmat M = L*trans(L);
//
//		M = eye<fmat>(dim,dim);
//
//
//		/* give singular value info */
//
//		fmat U;
//		fvec s;
//		fmat V;
//
//		svd(U,s,V,M);
//
//#if 0
//
//		printf("M = \n");
//		M.print();
//		printf("sigma = %10.7f\n",sigma);
//		printf("singular values of M = \n");
//		s.print();
//#endif
//
//
//
//		/* compute generalization error */
//
//		fmat XTest = dataTest.submat(0,0,number_of_test_samples-1,dim-1);
//		fvec yTest = dataTest.col(dim);
//
//
//#if 0
//		printf("number of test samples =%d\n",number_of_test_samples);
//		printf("XTest = \n");
//		XTest.print();
//		printf("yTest = \n");
//		trans(yTest).print();
//#endif
//		fmat XTraining = dataTraining.submat(0,0,number_of_training_samples-1,dim-1);
//		fvec yTraining = dataTraining.col(dim);
//
//#if 0
//		printf("dataTraining =\n");
//		dataTraining.print();
//		printf("yTraining =\n");
//		yTraining.print();
//#endif
//
//
//
//		fmat gradTraining;
//		if(number_of_samples_with_g_eval >0 ){
//
//			gradTraining = dataTraining.submat(0,dim+1,number_of_training_samples-1,2*dim);
//
//		}
//
//#if 0
//		printf("gradTraining =\n");
//		gradTraining.print();
//#endif
//
//
//
//
//		float genError = 0.0;
//		float genErrorFirstOrder = 0.0;
//
//		for(int i=0;i <number_of_test_samples; i++){
//
//			frowvec xp = XTest.row(i);
//			frowvec xpNotNormalized = XTestNotNormalized.row(i);
//#if 1
//			printf("xp = \n");
//			xp.print();
//			printf("ytest (normalized) = %10.7f\n",yTest(i)/yTrainingMax);
//
//#endif
//
//			float ytilde = 0.0;
//			float ytildefirst = 0.0;
//
//			if(number_of_samples_with_g_eval ==0 ) {
//
//				ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax ;
//
//			}
//			else{
//
//
//
//				ytilde = kernelRegressorNotNormalized(XTraining,
//						XTrainingNotNormalized,
//						yTrainingNotNormalized,
//						gradTrainingNotNormalized,
//						xpNotNormalized,
//						x_minTraining,
//						x_maxTraining,
//						M,
//						sigma);
//
//				/* this is the first order approximation */
//				ytildefirst = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax ;
//
//
//			}
//
//			float yexact = yTest(i);
//
//
//			genError += (yexact-ytilde)*(yexact-ytilde);
//			genErrorFirstOrder += (yexact-ytildefirst)*(yexact-ytildefirst);
//#if 1
//
//			printf("yexact = %10.7f, ytilde = %10.7f, ytildefirst = %10.7f\n", yexact,ytilde,ytildefirst);
//#endif
//
//
//			exit(1);
//		}
//
//		genError = genError/number_of_test_samples;
//		genErr1(iter) = genError;
//
//		genErrorFirstOrder = genErrorFirstOrder/number_of_test_samples;
//
//		genErr2(iter) = genErrorFirstOrder;
//
//		printf("genError (out of sample)= %10.7f\n",genError );
//		printf("genError (out of sample, without gradients)= %10.7f\n",genErrorFirstOrder );
//
//
//		/* generalization error for M = I */
//
//		M = eye<fmat>(dim,dim);
//		sigma = 0.1;
//
//		genError = 0.0;
//		genErrorFirstOrder = 0.0;
//
//		for(int i=0;i <number_of_test_samples; i++){
//
//
//			float ytilde = 0.0;
//			float ytildefirst = 0.0;
//
//			frowvec xp = XTest.row(i);
//			frowvec xpNotNormalized = XTestNotNormalized.row(i);
//			if(number_of_samples_with_g_eval ==0 ) {
//
//				ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax ;
//
//			}
//			else{
//
//
//
//				ytilde = kernelRegressorNotNormalized(XTraining,
//						XTrainingNotNormalized,
//						yTrainingNotNormalized,
//						gradTrainingNotNormalized,
//						xpNotNormalized,
//						x_minTraining,
//						x_maxTraining,
//						M,
//						sigma);
//
//
//				/* this is the first order approximation */
//				ytildefirst = kernelRegressor(XTraining, yTraining, xp, M, sigma)*yTrainingMax ;
//
//
//
//			}
//			float yexact = yTest(i);
//
//#if 1
//
//			printf("yexact = %10.7f, ytilde = %10.7f, ytildefirst = %10.7f\n", yexact,ytilde,ytildefirst);
//#endif
//
//			genError += (yexact-ytilde)*(yexact-ytilde);
//			genErrorFirstOrder += (yexact-ytildefirst)*(yexact-ytildefirst);
//
//
//		}
//
//		genError = genError/number_of_test_samples;
//		genErrorFirstOrder = genErrorFirstOrder/number_of_test_samples;
//
//		genErr3(iter) = genError;
//
//
//
//		genErr4(iter) = genErrorFirstOrder;
//
//
//		printf("genError (out of sample, M = I) = %10.7f\n",genError );
//		printf("genError (out of sample, M = I, without gradients) = %10.7f\n",genErrorFirstOrder );
//
//
//		float sum1 = sum(genErr1);
//		float sum2 = sum(genErr2);
//		float sum3 = sum(genErr3);
//		float sum4 = sum(genErr4);
//
//		float mean1 = sum1/(iter+1);
//		float mean2 = sum2/(iter+1);
//		float mean3 = sum3/(iter+1);
//		float mean4 = sum4/(iter+1);
//
//
//		printf("mean(genErr1) = %10.7f\n",mean1);
//		printf("mean(genErr2) = %10.7f\n",mean2);
//		printf("mean(genErr3) = %10.7f\n",mean3);
//		printf("mean(genErr4) = %10.7f\n",mean4);
//
//	}
//
//
//
//
//}
//
//
//
//
//void perform_aggregation_model_test_highdim(double (*test_function)(double *),
//		double (*test_function_adj)(double *, double *),
//		double *bounds,
//		std::string function_name,
//		int number_of_training_samples,
//		int sampling_method,
//		int dim){
//
//#if 1
//	printf("perform_aggregation_model_test_highdim ... \n");
//#endif
//
//	float sigma = 0.01;
//	float wSvd = 0.0;
//	float w12 = 1.0;
//
//	int number_of_trials = 100;
//
//	std::string datafilename = function_name+".csv";
//	std::string validationfilename = function_name+"_"+ std::to_string(number_of_training_samples)+"_validation.csv";
//
//
//
//
//	unsigned int number_of_test_samples = 10000;
//
//	srand (time(NULL));
//
//
//	vec genErr1(number_of_trials); genErr1.fill(0.0);
//	vec genErr2(number_of_trials); genErr2.fill(0.0);
//	vec genErr3(number_of_trials); genErr3.fill(0.0);
//
//
//	for(int iter=0; iter<number_of_trials; iter++){
//
//#if 1
//		printf("outer iteration = %d\n",iter);
//#endif
//
//
//
//#if 1
//		printf("Generating validation data with %d samples...\n", number_of_test_samples);
//#endif
//		generate_highdim_test_function_data_cuda(test_function,
//				validationfilename,
//				bounds,
//				number_of_test_samples,
//				dim);
//
//		/* generate data with gradients */
//#if 1
//		printf("Generating training data with %d samples...\n", number_of_training_samples);
//#endif
//		generate_highdim_test_function_data_GEK(test_function,
//				test_function_adj,
//				datafilename,
//				bounds,
//				dim,
//				0,
//				number_of_training_samples,
//				sampling_method);
//
//		/* set up the aggregation model */
//
//		AggregationModel settingsAggModel(function_name,dim);
//
//		settingsAggModel.validationset_input_filename = validationfilename;
//		settingsAggModel.max_number_of_kriging_iterations = 10000;
//		settingsAggModel.number_of_cv_iterations = 20;
//
//
//		/* set visualization settings */
//		settingsAggModel.visualizeKrigingValidation = "yes";
//		settingsAggModel.visualizeKernelRegressionValidation = "yes";
//		settingsAggModel.visualizeAggModelValidation = "yes";
//
//		settingsAggModel.train();
//
//		exit(1);
//
//
//		genErr1(iter)= settingsAggModel.genErrorKriging;
//		genErr2(iter)= settingsAggModel.genErrorKernelRegression;
//		genErr3(iter)= settingsAggModel.genErrorAggModel;
//
//
//
//		float sum1 = sum(genErr1);
//		float sum2 = sum(genErr2);
//		float sum3 = sum(genErr3);
//
//		float mean1 = sum1/(iter+1);
//		float mean2 = sum2/(iter+1);
//		float mean3 = sum3/(iter+1);
//
//		printf("MSE,Kriging       = %10.7f, MSE,KerReg       = %10.7f, MSE,AggMod       = %10.7f\n",genErr1(iter),genErr2(iter),genErr3(iter));
//		printf("mean(MSE,Kriging) = %10.7f, mean(MSE,KerReg) = %10.7f, mean(MSE,AggMod) = %10.7f\n",mean1,mean2,mean3);
//
//
//	}
//
//
//
//
//}






