/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <stack>
#include <cassert>
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "auxiliary_functions.hpp"
#include "test_functions.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"

#include "optimization.hpp"
#include "random_functions.hpp"
#include "lhs.hpp"
#include "bounds.hpp"



#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
using namespace arma;




TestFunction::TestFunction(std::string name,int dim):boxConstraints(dim){

	assert(dim > 0);
	assert(!name.empty());

	dimension = dim;
	function_name = name;


}




void TestFunction::setBoxConstraints(double lb, double ub){

	assert(dimension>0);
	Bounds parameterBounds(dimension);
	parameterBounds.setBounds(lb, ub);

	boxConstraints = parameterBounds;


}


void TestFunction::evaluateGlobalExtrema(void) const{


	rowvec maxx(dimension);
	rowvec minx(dimension);

	double globalMin = LARGE;
	double globalMax = -LARGE;



	const int numberOfBruteForceIterations = 100000;

	for(unsigned int i=0; i<numberOfBruteForceIterations; i++ ){

		rowvec x = boxConstraints.generateVectorWithinBounds();
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

	printf("\nPrinting function information...\n");
	printf("Function name = %s\n",function_name.c_str());
	printf("Number of independent variables = %d\n",dimension);
	printf("Parameter bounds:\n");


	printMatrix(trainingSamples, "trainingSamples");
	printMatrix(testSamples, "testSamples");
	boxConstraints.print();


}

void TestFunction::evaluate(Design &d) const{

	rowvec x= d.designParameters;
	if(evaluationSelect == 1){

		assert(func_ptr!=NULL);
		d.trueValue = func_ptr(x.memptr());

	}

	if(evaluationSelect == 2){

		assert(adj_ptr!=NULL);
		d.trueValue = adj_ptr(x.memptr(),d.gradient.memptr());

	}

	if(evaluationSelect == 3){

		assert(tan_ptr!=NULL);
		double fdot;
		d.trueValue = tan_ptr(x.memptr(), d.tangentDirection.memptr(), &fdot);
		d.tangentValue = fdot;

	}

	if(evaluationSelect == 4){

		assert(func_ptrLowFi!=NULL);
		d.trueValue = func_ptrLowFi(x.memptr());

	}
	if(evaluationSelect == 5){

		assert(adj_ptrLowFi!=NULL);
		d.trueValue = adj_ptrLowFi(x.memptr(),d.gradient.memptr());

	}
	if(evaluationSelect == 6){

		assert(tan_ptrLowFi!=NULL);
		double fdot;
		d.trueValue = tan_ptrLowFi(x.memptr(), d.tangentDirection.memptr(), &fdot);
		d.tangentValue = fdot;

	}


}


void TestFunction::generateSamplesInputTrainingData(void){

	assert(numberOfTrainingSamples> 0);
	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();

	unsigned int howManySamples;
	if(numberOfTrainingSamplesLowFi > 0){

		howManySamples = numberOfTrainingSamplesLowFi;
	}
	else{

		howManySamples = numberOfTrainingSamples;
	}

	LHSSamples samplesTraining(dimension, lb, ub, howManySamples);


	trainingSamplesInput = samplesTraining.getSamples();
	trainingSamplesInput = shuffleRows(trainingSamplesInput);

	ifInputSamplesAreGenerated = true;


}

void TestFunction::generateSamplesInputTestData(void){

	assert(numberOfTestSamples> 0);
	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();
	LHSSamples samplesTest(dimension, lb, ub, numberOfTestSamples);

	testSamplesInput = samplesTest.getSamples();

	if(dimension == 1) testSamplesInput = sort(testSamplesInput);

}

mat TestFunction::generateSamplesWithFunctionalValues(mat input, unsigned int N) const{

	assert(input.n_rows >= N);

	mat samples(N, dimension+1, fill::zeros);

	for (unsigned int i = 0; i < N; i++) {
		rowvec dv = input.row(i);
		Design d(dv);
		evaluate(d);
		rowvec sample(dimension + 1);
		copyRowVector(sample, dv);
		sample(dimension) = d.trueValue;
		samples.row(i) = sample;
	}

	return samples;
}

mat TestFunction::generateSamplesWithAdjoints(mat input, unsigned int N) const {

	assert(input.n_rows >= N);

	mat samples(N, 2*dimension+1, fill::zeros);

	for (unsigned int i = 0; i < N; i++) {
		rowvec dv = input.row(i);
		Design d(dv);
		evaluate(d);
		rowvec sample(2*dimension +1);
		copyRowVector(sample,dv);
		sample(dimension) = d.trueValue;
		copyRowVector(sample,d.gradient,dimension+1);
		samples.row(i) = sample;
	}

	return samples;
}


mat TestFunction::generateSamplesWithTangents(mat input, unsigned int N) const{

	assert(input.n_rows >= N);

	mat trainingSamplesTangentDirections(N,dimension);
	trainingSamplesTangentDirections.randu();
	mat samples(N, 2*dimension+2, fill::zeros);

	for (unsigned int i = 0; i < N; i++) {

		rowvec dv = input.row(i);
		Design d(dv);

		rowvec dir = trainingSamplesTangentDirections.row(i);
		dir = makeUnitVector(dir);
		d.tangentDirection = dir;
		evaluate(d);
		rowvec sample(2*dimension + 2);
		copyRowVector(sample,dv);
		sample(dimension)   = d.trueValue;
		sample(dimension+1) = d.tangentValue;
		copyRowVector(sample,d.tangentDirection,dimension+2);
		samples.row(i) = sample;

	}

	return samples;
}



void TestFunction::generateTrainingSamples(void){

	assert(isNotEmpty(filenameTrainingData));
	assert(boxConstraints.areBoundsSet());

	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}



	evaluationSelect = 1;
	trainingSamples = generateSamplesWithFunctionalValues(trainingSamplesInput, numberOfTrainingSamples);

	saveMatToCVSFile(trainingSamples, filenameTrainingData);
}


void TestFunction::generateTrainingSamplesWithAdjoints(void){

	assert(isNotEmpty(filenameTrainingData));
	assert(boxConstraints.areBoundsSet());

	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 2;
	trainingSamples = generateSamplesWithAdjoints(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingData);
}


void TestFunction::generateTrainingSamplesMultiFidelity(void){

	assert(isNotEmpty(filenameTrainingDataHighFidelity));
	assert(isNotEmpty(filenameTrainingDataLowFidelity));
	assert(boxConstraints.areBoundsSet());
	assert(numberOfTrainingSamplesLowFi > numberOfTrainingSamples);

	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}

	evaluationSelect = 1;
	trainingSamples = generateSamplesWithFunctionalValues(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingDataHighFidelity);

	evaluationSelect = 4;
	trainingSamplesLowFidelity = generateSamplesWithFunctionalValues(trainingSamplesInput,numberOfTrainingSamplesLowFi);
	saveMatToCVSFile(trainingSamplesLowFidelity, filenameTrainingDataLowFidelity);

}


void TestFunction::generateTrainingSamplesMultiFidelityWithAdjoint(void){

	assert(isNotEmpty(filenameTrainingDataHighFidelity));
	assert(isNotEmpty(filenameTrainingDataLowFidelity));
	assert(boxConstraints.areBoundsSet());
	assert(numberOfTrainingSamplesLowFi > numberOfTrainingSamples);


	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 2;
	trainingSamples = generateSamplesWithAdjoints(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingDataHighFidelity);

	evaluationSelect = 5;
	trainingSamplesLowFidelity = generateSamplesWithAdjoints(trainingSamplesInput,numberOfTrainingSamplesLowFi);
	saveMatToCVSFile(trainingSamplesLowFidelity, filenameTrainingDataLowFidelity);

}


void TestFunction::generateTrainingSamplesWithTangents(void){

	assert(tan_ptr!=NULL);
	assert(isNotEmpty(filenameTrainingData));
	assert(boxConstraints.areBoundsSet());

	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 3;
	trainingSamples = generateSamplesWithTangents(trainingSamplesInput,numberOfTrainingSamples);

	saveMatToCVSFile(trainingSamples, filenameTrainingData);

}

void TestFunction::generateTrainingSamplesMultiFidelityWithTangents(void){

	assert(isNotEmpty(filenameTrainingDataHighFidelity));
	assert(isNotEmpty(filenameTrainingDataLowFidelity));
	assert(boxConstraints.areBoundsSet());
	assert(numberOfTrainingSamplesLowFi > numberOfTrainingSamples);


	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 3;

	trainingSamples = generateSamplesWithTangents(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingDataHighFidelity);

	evaluationSelect = 6;
	trainingSamplesLowFidelity = generateSamplesWithTangents(trainingSamplesInput,numberOfTrainingSamplesLowFi);
	saveMatToCVSFile(trainingSamplesLowFidelity, filenameTrainingDataLowFidelity);

}


void TestFunction::generateTrainingSamplesMultiFidelityWithLowFiAdjoint(void){

	assert(isNotEmpty(filenameTrainingDataHighFidelity));
	assert(isNotEmpty(filenameTrainingDataLowFidelity));
	assert(boxConstraints.areBoundsSet());
	assert(numberOfTrainingSamplesLowFi > numberOfTrainingSamples);


	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 1;
	trainingSamples = this->generateSamplesWithFunctionalValues(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingDataHighFidelity);

	evaluationSelect = 5;
	trainingSamplesLowFidelity = generateSamplesWithAdjoints(trainingSamplesInput,numberOfTrainingSamplesLowFi);
	saveMatToCVSFile(trainingSamplesLowFidelity, filenameTrainingDataLowFidelity);

}

void TestFunction::generateTrainingSamplesMultiFidelityWithLowFiTangents(void){

	assert(isNotEmpty(filenameTrainingDataHighFidelity));
	assert(isNotEmpty(filenameTrainingDataLowFidelity));
	assert(boxConstraints.areBoundsSet());
	assert(numberOfTrainingSamplesLowFi > numberOfTrainingSamples);


	if(!ifInputSamplesAreGenerated){

		generateSamplesInputTrainingData();
	}
	evaluationSelect = 1;
	trainingSamples = generateSamplesWithFunctionalValues(trainingSamplesInput,numberOfTrainingSamples);
	saveMatToCVSFile(trainingSamples, filenameTrainingDataHighFidelity);

	evaluationSelect = 6;
	trainingSamplesLowFidelity = generateSamplesWithTangents(trainingSamplesInput,numberOfTrainingSamplesLowFi);
	saveMatToCVSFile(trainingSamplesLowFidelity, filenameTrainingDataLowFidelity);

}



void TestFunction::generateTestSamples(void){

	assert(isNotEmpty(filenameTestData));

	evaluationSelect = 1;
	generateSamplesInputTestData();
	testSamples = generateSamplesWithFunctionalValues(testSamplesInput,numberOfTestSamples);
	saveMatToCVSFile(testSamples, filenameTestData);

}

















double Waves2DWithHighFrequencyPart(double *x){

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

double Waves2DWithHighFrequencyPartAdj(double *x,double *xb){
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
/********************************************************************************/
double testFunction1D(double *x){

	double c1,c2,c3,c4,c5;
	c1 = 1.0;
	c2 = 5.0;
	c3 = 5.0;
	c4 = 2.0;
	c5 = 4.0;
	return exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;

}
double testFunction1DAdj(double *x, double *xb) {

	double c1,c2,c3,c4,c5;
	c1 = 1.0;
	c2 = 5.0;
	c3 = 5.0;
	c4 = 2.0;
	c5 = 4.0;
	xb[0] = -c1*exp(-c1*x[0]) + c2*cos(c2*x[0]) - c3*sin(c3*x[0]) + c4;
	return exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;
}

double testFunction1DTangent(double *x, double *xd, double *fdot) {
	double c1, c2, c3, c4, c5;
	c1 = 1.0;
	c2 = 5.0;
	c3 = 5.0;
	c4 = 2.0;
	c5 = 4.0;
	double f = exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;
	double fd = (cos(c2*x[0])*c2-exp(-(c1*x[0]))*c1-sin(c3*x[0])*c3+c4)*xd[0];
	*fdot = fd;
	return f;
}

double testFunction1DLowFi(double *x){

	double c1,c2,c3,c4,c5;
	c1 = 1.12;
	c2 = 4.96;
	c3 = 5.1;
	c4 = 2.3;
	c5 = 3.89;
	return exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;

}

double testFunction1DAdjLowFi(double *x, double *xb) {

	double c1,c2,c3,c4,c5;
	c1 = 1.12;
	c2 = 4.96;
	c3 = 5.1;
	c4 = 2.3;
	c5 = 3.89;
	xb[0] = -c1*exp(-c1*x[0]) + c2*cos(c2*x[0]) - c3*sin(c3*x[0]) + c4;
	return exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;
}

double testFunction1DTangentLowFi(double *x, double *xd, double *fdot) {
	double c1, c2, c3, c4, c5;
	c1 = 1.12;
	c2 = 4.96;
	c3 = 5.1;
	c4 = 2.3;
	c5 = 3.89;
	double f = exp(-c1*x[0]) + sin(c2*x[0]) + cos(c3*x[0]) + c4*x[0] + c5;
	double fd = (cos(c2*x[0])*c2-exp(-(c1*x[0]))*c1-sin(c3*x[0])*c3+c4)*xd[0];
	*fdot = fd;
	return f;
}




/**************************************************************************************/
double LinearTF1(double *x){
	return 2*x[0]+3*x[1]+1.5;
}

double LinearTF1LowFidelity(double *x){
	return 2.2*x[0]+2.9*x[1]+1.2 ;
}

double LinearTF1LowFidelityAdj(double *x, double *xb){
	xb[0] = xb[0] + 2.2;
	xb[1] = xb[1] + 2.9;
	return 2.2*x[0]+2.9*x[1]+1.2;
}

double LinearTF1Adj(double *x, double *xb) {
	xb[0] = xb[0] + 2.0;
	xb[1] = xb[1] + 3.0;
	return 2*x[0]+3*x[1]+1.5;
}
double LinearTF1Tangent(double *x, double *xd, double *fdot) {
	*fdot = 2.0*xd[0] + 3.0*xd[1];
	return 2*x[0]+3*x[1]+1.5;
}
double LinearTF1LowFidelityTangent(double *x, double *xd, double *fdot) {
	*fdot = 2.2*xd[0] + 2.9*xd[1];
	return 2.2*x[0]+2.9*x[1]+1.5;
}

/**************************************************************************************/


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

	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );


}


double HimmelblauTangent(double *x, double *xd, double *fdot) {
	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	double f;
	double fd;
	double arg1;
	double arg1d;
	double arg2;
	double arg2d;
	arg1d = c1*2*x[0]*xd[0] + c2*xd[1];
	arg1 = c1*x[0]*x[0] + c2*x[1] - c3;
	arg2d = c4*xd[0] + c5*2*x[1]*xd[1];
	arg2 = c4*x[0] + c5*x[1]*x[1] - c6;
	fd = 2.0*pow(arg1, 2.0-1)*arg1d + 2.0*pow(arg2, 2.0-1)*arg2d;
	f = pow(arg1, 2.0) + pow(arg2, 2.0);
	*fdot = fd;
	return f;
}


double HimmelblauAdj(double *x, double *xb) {
	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	double f = 0.0;
	double fb = 0.0;
	double tempb;
	double tempb0;
	double Himmelblau;
	fb = 1.0;
	tempb = 2.0*pow(c1*(x[0]*x[0])-c3+c2*x[1], 2.0-1)*fb;
	tempb0 = 2.0*pow(c4*x[0]-c6+c5*(x[1]*x[1]), 2.0-1)*fb;
	xb[0] = xb[0] + c4*tempb0 + 2*x[0]*c1*tempb;
	xb[1] = xb[1] + 2*x[1]*c5*tempb0 + c2*tempb;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );

}

double HimmelblauAdjLowFi(double *x, double *xb) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	double f = 0.0;
	double fb = 0.0;
	double tempb;
	double tempb0;
	double Himmelblau;
	fb = 1.0;
	tempb = 2.0*pow(c1*(x[0]*x[0])-c3+c2*x[1], 2.0-1)*fb;
	tempb0 = 2.0*pow(c4*x[0]-c6+c5*(x[1]*x[1]), 2.0-1)*fb;
	xb[0] = xb[0] + c4*tempb0 + 2*x[0]*c1*tempb;
	xb[1] = xb[1] + 2*x[1]*c5*tempb0 + c2*tempb;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );

}


double HimmelblauLowFi(double *x){

	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );


}

double HimmelblauTangentLowFi(double *x, double *xd, double *fdot) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	double f;
	double fd;
	double arg1;
	double arg1d;
	double arg2;
	double arg2d;
	arg1d = c1*2*x[0]*xd[0] + c2*xd[1];
	arg1 = c1*x[0]*x[0] + c2*x[1] - c3;
	arg2d = c4*xd[0] + c5*2*x[1]*xd[1];
	arg2 = c4*x[0] + c5*x[1]*x[1] - c6;
	fd = 2.0*pow(arg1, 2.0-1)*arg1d + 2.0*pow(arg2, 2.0-1)*arg2d;
	f = pow(arg1, 2.0) + pow(arg2, 2.0);
	*fdot = fd;
	return f;
}

/***********************************************************************************/



double himmelblauConstraintFunction1(double *x){

	return x[0]*x[0]+ x[1]*x[1];

}


double himmelblauConstraintFunction2(double *x){

	return x[0]+ x[1];

}







void generateEggholderData(std::string filename, unsigned int nSamples){


	mat samples(nSamples,3);

	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		double xInp[2];
		x(0) = generateRandomDouble(0.0,512.0);
		x(1) = generateRandomDouble(0.0,512.0);
		xInp[0] = x(0);
		xInp[1] = x(1);

		x(2) = Eggholder(xInp);
		samples.row(i) = x;

	}


	saveMatToCVSFile(samples,filename);

}

void generateEggholderDataMultiFidelity(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi){

	assert(nSamplesHiFi < nSamplesLowFi);
	assert(filenameHiFi != filenameLowFi);
	mat samplesLowFi(nSamplesLowFi,3);

	for (unsigned int i=0; i<nSamplesLowFi; i++){
		rowvec x(3);
		double xInp[2];
		x(0) = generateRandomDouble(0.0,512.0);
		x(1) = generateRandomDouble(0.0,512.0);
		xInp[0] = x(0);
		xInp[1] = x(1);

		x(2) = Eggholder(xInp) + generateRandomDouble(-10.0,10.0);
		samplesLowFi.row(i) = x;

	}


	saveMatToCVSFile(samplesLowFi, filenameLowFi);

	mat samplesHiFi(nSamplesHiFi,3);
	for (unsigned int i=0; i<nSamplesHiFi; i++){

		double xInp[2];
		xInp[0] = samplesLowFi(i,0);
		xInp[1] = samplesLowFi(i,1);
		samplesHiFi(i,0) = samplesLowFi(i,0);
		samplesHiFi(i,1) = samplesLowFi(i,1);
		samplesHiFi(i,2) = Eggholder(xInp);

	}

	saveMatToCVSFile(samplesHiFi, filenameHiFi);
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

double Eggholder(vec x){
	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));

}

double EggholderAdj(double *x, double *xb) {
	double fabs0;
	double fabs0b;
	double fabs1;
	double fabs1b;
	double temp;
	double temp0;
	int branch;

	xb[0] = 0.0;
	xb[1] = 0.0;

	std::stack<int> intStack;


	if (x[1] + 0.5*x[0] + 47.0 >= 0.0) {
		fabs0 = x[1] + 0.5*x[0] + 47.0;
		intStack.push(1);
	} else {
		fabs0 = -(x[1]+0.5*x[0]+47.0);
		intStack.push(0);
	}
	if (x[0] - (x[1] + 47.0) >= 0.0) {
		fabs1 = x[0] - (x[1] + 47.0);
		intStack.push(0);
	} else {
		fabs1 = -(x[0]-(x[1]+47.0));
		intStack.push(1);
	}
	temp = sqrt(fabs0);
	temp0 = sqrt(fabs1);
	xb[1] = xb[1] - sin(temp);
	fabs0b = (fabs0 == 0.0 ? 0.0 : -(cos(temp)*(x[1]+47.0)/(2.0*
			temp)));
	xb[0] = xb[0] - sin(temp0);
	fabs1b = (fabs1 == 0.0 ? 0.0 : -(cos(temp0)*x[0]/(2.0*temp0)));

	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] + fabs1b;
		xb[1] = xb[1] - fabs1b;
	} else {
		xb[1] = xb[1] + fabs1b;
		xb[0] = xb[0] - fabs1b;
	}
	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] - 0.5*fabs0b;
		xb[1] = xb[1] - fabs0b;
	} else {
		xb[1] = xb[1] + fabs0b;
		xb[0] = xb[0] + 0.5*fabs0b;
	}

	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));
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

double McCormickAdj(double *x, double *xb) {
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

double GoldsteinPrice(double *x){
	double x_sqr  =  x[0]*x[0]; // x^2
	double y_sqr  =  x[1]*x[1]; // y^2
	double xy     =  x[0]*x[1]; // xy
	double temp1  = pow( (x[0]+x[1]+1), 2.0);
	double temp2  = (19.0-14.0*x[0]+3.0*x_sqr-14.0*x[1]+6.0*xy+ 3.0*y_sqr);
	double temp3  = pow( (2*x[0]-3.0*x[1]), 2.0);
	double temp4  = (18.0 - 32.0 *x[0] + 12.0* x_sqr +48.0*x[1]-36.0*xy+27.0*y_sqr);

	return (1.0+temp1*temp2)*(30.0+temp3*temp4);

}


double GoldsteinPriceAdj(double *x, double *xb) {
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

double RosenbrockAdj(double *x, double *xb) {
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

double Rosenbrock3DAdj(double *x, double *xb) {

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

double Rosenbrock4DAdj(double *x, double *xb) {
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


double Rosenbrock8DAdj(double *x, double *xb) {
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


double Shubertadj(double *x, double *xb) {
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






double Himmelblau(vec x){

	return Himmelblau(x.memptr());

}

double HimmelblauLowFi(vec x){

	return Himmelblau(x.memptr()) + Waves2D(x.memptr())*50.0;

}



vec HimmelblauGradient(vec x){

	vec gradient(2);
	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	gradient[0] = tempb0 + 2*x[0]*tempb;
	gradient[1] = 2*x[1]*tempb0 + tempb;

	return gradient;
}


void generateHimmelblauDataMultiFidelity(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi){

	assert(nSamplesHiFi < nSamplesLowFi);
	assert(filenameHiFi != filenameLowFi);
	mat samplesLowFi(nSamplesLowFi,3);

	for (unsigned int i=0; i<nSamplesLowFi; i++){
		rowvec x(3);
		double xInp[2];
		x(0) = generateRandomDouble(-6.0,6.0);
		x(1) = generateRandomDouble(-6.0,6.0);
		xInp[0] = x(0);
		xInp[1] = x(1);
		double fVal = Himmelblau(xInp);
		double errorTerm = Waves2D(xInp)*50.0;
#if 0
		std::cout<<"fVal = "<<fVal<<"\n";
		std::cout<<"errorTerm = "<<errorTerm<<"\n\n";

#endif
		x(2) = fVal + errorTerm;
		samplesLowFi.row(i) = x;

	}

	saveMatToCVSFile(samplesLowFi, filenameLowFi);

	mat samplesHiFi(nSamplesHiFi,3);
	for (unsigned int i=0; i<nSamplesHiFi; i++){

		double xInp[2];
		xInp[0] = samplesLowFi(i,0);
		xInp[1] = samplesLowFi(i,1);
		samplesHiFi(i,0) = samplesLowFi(i,0);
		samplesHiFi(i,1) = samplesLowFi(i,1);
		samplesHiFi(i,2) = Himmelblau(xInp);

	}

	saveMatToCVSFile(samplesHiFi, filenameHiFi);

}


void generateHimmelblauDataMultiFidelityWithShuffle(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi){

	assert(nSamplesHiFi < nSamplesLowFi);
	assert(filenameHiFi != filenameLowFi);
	mat samplesLowFi(nSamplesLowFi,3);

	for (unsigned int i=0; i<nSamplesLowFi; i++){
		rowvec x(3);
		double xInp[2];
		x(0) = generateRandomDouble(-6.0,6.0);
		x(1) = generateRandomDouble(-6.0,6.0);
		xInp[0] = x(0);
		xInp[1] = x(1);
		double fVal = Himmelblau(xInp);
		double errorTerm = Waves2D(xInp)*50.0;
#if 0
		std::cout<<"fVal = "<<fVal<<"\n";
		std::cout<<"errorTerm = "<<errorTerm<<"\n\n";

#endif
		x(2) = fVal + errorTerm;
		samplesLowFi.row(i) = x;

	}

	saveMatToCVSFile(samplesLowFi, filenameLowFi);

	mat samplesHiFi(nSamplesHiFi,3);
	for (unsigned int i=0; i<nSamplesHiFi; i++){

		double xInp[2];
		xInp[0] = samplesLowFi(i,0);
		xInp[1] = samplesLowFi(i,1);
		samplesHiFi(i,0) = samplesLowFi(i,0);
		samplesHiFi(i,1) = samplesLowFi(i,1);
		samplesHiFi(i,2) = Himmelblau(xInp);

	}

	samplesHiFi = shuffleRows(samplesHiFi);
	saveMatToCVSFile(samplesHiFi, filenameHiFi);

}





void generateHimmelblauDataMultiFidelityWithGradients(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi){

	assert(nSamplesHiFi < nSamplesLowFi);
	assert(filenameHiFi != filenameLowFi);
	mat samplesLowFi(nSamplesLowFi,5);

	for (unsigned int i=0; i<nSamplesLowFi; i++){
		rowvec x(5);
		double xInp[2];
		double xInpb[2];
		x(0) = generateRandomDouble(-6.0,6.0);
		x(1) = generateRandomDouble(-6.0,6.0);
		xInp[0] = x(0);
		xInp[1] = x(1);
		double fVal = HimmelblauAdj(xInp,xInpb);
		double xInpbErr[2];
		double errorTerm = Waves2DAdj(xInp,xInpbErr)*50.0;

		x(2) = fVal + errorTerm;
		x(3) = xInpb[0] + 50.0*xInpbErr[0];
		x(4) = xInpb[1] + 50.0*xInpbErr[1];
		samplesLowFi.row(i) = x;

	}


	saveMatToCVSFile(samplesLowFi, filenameLowFi);

	mat samplesHiFi(nSamplesHiFi,5);
	for (unsigned int i=0; i<nSamplesHiFi; i++){

		double xInp[2];

		xInp[0] = samplesLowFi(i,0);
		xInp[1] = samplesLowFi(i,1);
		double fVal = Himmelblau(xInp);
		samplesHiFi(i,0) = samplesLowFi(i,0);
		samplesHiFi(i,1) = samplesLowFi(i,1);
		samplesHiFi(i,2) = fVal;
		samplesHiFi(i,3) = samplesLowFi(i,3);
		samplesHiFi(i,4) = samplesLowFi(i,4);

	}

	saveMatToCVSFile(samplesHiFi, filenameHiFi);
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
















