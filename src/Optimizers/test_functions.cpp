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
 * General Public License along with RoDEO.
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


#include "optimization.hpp"
#include "random_functions.hpp"
#include "bounds.hpp"
#include "matrix_operations.hpp"
#include "vector_operations.hpp"


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

void TestFunction::setBoxConstraints(Bounds bounds){
	assert(dimension>0);
	assert(bounds.areBoundsSet());
	assert(bounds.getDimension() == dimension);

	boxConstraints = bounds;
}

pair<double,double> TestFunction::evaluateGlobalExtrema(void) const{

	assert(dimension>0);

	vec maxx(dimension);
	vec minx(dimension);

	double globalMin = LARGE;
	double globalMax = -LARGE;


	for(int i=0; i<numberOfBruteForceIterations; i++ ){


		vec x = boxConstraints.generateVectorWithinBounds();
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
	trans(minx).print();
	printf("Function value at maximum = %10.7f\n",globalMin);
	printf("Function has maximum at x:\n");
	trans(maxx).print();
	printf("Function value at maximum = %10.7f\n",globalMax);

	pair<double,double> extrema;
	extrema.first  = globalMin;
	extrema.second = globalMax;

	return extrema;

}


void TestFunction::print(void){

	printf("\nPrinting function information...\n");
	printf("Function name = %s\n",function_name.c_str());
	printf("Number of independent variables = %d\n",dimension);
	printf("Parameter bounds:\n");
	trainingSamples.print("trainingSamples");
	testSamples.print("testSamples");
	boxConstraints.print();


}



void TestFunction::evaluate(Design &d) const{

	rowvec x= d.designParameters;
	assert(x.size() == dimension);


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
		assert(d.tangentDirection.size() == dimension);

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
	assert(boxConstraints.areBoundsSet());

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();

	unsigned int howManySamples;
	if(numberOfTrainingSamplesLowFi > 0){

		howManySamples = numberOfTrainingSamplesLowFi;
	}
	else{

		howManySamples = numberOfTrainingSamples;
	}

	trainingSamplesInput = generateRandomMatrix(howManySamples,lb,ub);
	ifInputSamplesAreGenerated = true;


}

void TestFunction::generateSamplesInputTestData(void){

	assert(numberOfTestSamples> 0);
	assert(boxConstraints.areBoundsSet());

	vec lb = boxConstraints.getLowerBounds();
	vec ub = boxConstraints.getUpperBounds();

//
//	LHSSamples samplesTest(dimension, lb, ub, numberOfTestSamples);
//	testSamplesInput = samplesTest.getSamples();


	testSamplesInput =  generateRandomMatrix(numberOfTestSamples,lb,ub);


	if(dimension == 1) testSamplesInput = sort(testSamplesInput);

}

void TestFunction::generateSamplesInputTestDataCloseToTrainingSamples(void){

	assert(numberOfTestSamples> 0);
	assert(trainingSamplesInput.n_rows > 0);
	testSamplesInput = trainingSamplesInput;
	numberOfTestSamples = trainingSamplesInput.n_rows;

	for(unsigned int i=0; i<trainingSamplesInput.n_rows; i++)
		for(unsigned int j=0; j<trainingSamplesInput.n_cols; j++) {

			double epsilon = trainingSamplesInput(i,j)*0.001;
			double perturbation = epsilon*generateRandomDouble(-1.0,1.0);
			testSamplesInput(i,j)= trainingSamplesInput(i,j)+perturbation;
		}

	testSamplesInput.print("test samples");

}



mat TestFunction::generateSamplesWithFunctionalValues(mat input, unsigned int N) const{

	assert(input.n_rows >= N);

	mat samples(N, dimension+1, fill::zeros);

	for (unsigned int i = 0; i < N; i++) {
		rowvec dv = input.row(i);
		Design d(dv);
		evaluate(d);
		rowvec sample(dimension + 1);
		copyVector(sample, dv);
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

		if(ifSomeAdjointsAreLeftBlank){

			if(i%2 == 0) d.gradient.fill(0.0);

		}

		rowvec sample(2*dimension +1);
		copyVector(sample,dv);
		sample(dimension) = d.trueValue;
		copyVector(sample,d.gradient,dimension+1);
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
		copyVector(sample,dv);
		sample(dimension)   = d.trueValue;

		if(ifSomeDirectionalDerivativesAreLeftBlank){
			if(i%2 == 0) {
				d.tangentValue = 0.0;
				d.tangentDirection.fill(0.0);
			}

		}

		sample(dimension+1) = d.tangentValue;
		copyVector(sample,d.tangentDirection,dimension+2);
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

void TestFunction::generateTestSamplesCloseToTrainingSamples(void){

	assert(isNotEmpty(filenameTestData));

	evaluationSelect = 1;
	generateSamplesInputTestDataCloseToTrainingSamples();
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






double himmelblauConstraintFunction1(double *x){

	return x[0]*x[0]+ x[1]*x[1];

}


double himmelblauConstraintFunction2(double *x){

	return x[0]+ x[1];

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





double Alpine02_5D(double *x){

	double prod = 1.0;
	for(unsigned int i=0; i<5; i++){
		prod = prod * sqrt(x[i])*sin(x[i]);
	}
	return prod;
}

double Alpine02_5DTangent(double *x, double *xd, double *fdot) {
	double prod = 1.0;
	double prodd;
	double result1;
	double result1d;
	prodd = 0.0;

	double temp;
	for (unsigned int i = 0; i < 5; ++i) {
		temp = sqrt(x[i]);
		result1d = (x[i] == 0.0 ? 0.0 : xd[i]/(2.0*temp));
		result1 = temp;
		temp = sin(x[i]);
		prodd = temp*(result1*prodd+prod*result1d) + prod*result1*cos(x[i])*
				xd[i];
		prod = prod*result1*temp;
	}

	*fdot = prodd;
	return prod;
}



double Alpine02_5DAdj(double *x, double *xb) {

	double prod  = 1.0;
	double prodb = 1.0;

	stack<double> realStack;

	for (int i = 0; i < 5; ++i) {
		xb[i] = 0.0;
	}

	for (int i = 0; i < 5; ++i) {
		realStack.push(prod);
		prod = prod*sqrt(x[i])*sin(x[i]);
	}

	double tempb;
	double temp;

	for (int i = 4; i > -1; --i) {

		prod =  realStack.top();
		realStack.pop();
		temp = sqrt(x[i]);
		tempb = sin(x[i])*prodb;
		if (x[i] == 0.0)
			xb[i] = xb[i] + cos(x[i])*prod*temp*prodb;
		else
			xb[i] = xb[i] + cos(x[i])*prod*temp*prodb + prod*tempb/(2.0*temp);

		prodb = temp*tempb;
	}

	return Alpine02_5D(x);
}










