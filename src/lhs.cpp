/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "lhs.hpp"
#include "random_functions.hpp"
#include "matrix_vector_operations.hpp"
#include <cassert>


LHSSamples::LHSSamples(unsigned int d, double lb, double ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	assert(ub>lb);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);
	upperBounds.fill(ub);
	lowerBounds.fill(lb);

	samples = zeros<mat>(N,d);


	generateSamples();

}

LHSSamples::LHSSamples(unsigned int d, double *lb, double *ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	for(unsigned int i=0; i<d; i++){

		assert(ub[i]>lb[i]);
	}


	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);

	for(unsigned int i=0; i<d; i++){
		upperBounds(i) = ub[i];
		lowerBounds(i) = lb[i];
	}

	samples = zeros<mat>(N,d);
	generateSamples();

}

LHSSamples::LHSSamples(unsigned int d, vec lb, vec ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	for(unsigned int i=0; i<d; i++){

		assert(ub(i)>lb(i));
	}


	upperBounds= ub;
	lowerBounds= lb;

	samples = zeros<mat>(N,d);

	generateSamples();


}


void LHSSamples::setDiscreteParameterIndices(int *indices, int size){

	for(unsigned int i=0; i<size; i++) {

		indicesDiscreteVariables.push_back(indices[i]);
	}


}

void LHSSamples::setDiscreteParameterIncrements(vec increments){

	incrementsDiscreteVariables = increments;

}


uvec LHSSamples::returnValidIntervalsForADimension(mat validIntervals, unsigned int dim){

	assert(dim>=0);
	assert(dim<numberOfDesignVariables);

	// first count how many valid intervals

	int count = 0;

	for(unsigned int i=0; i<validIntervals.n_cols; i++){

		if(validIntervals(dim,i) > -1) count++;

	}

	uvec validIntervalsForADimension(count);

	count = 0;
	for(unsigned int i=0; i<validIntervals.n_cols; i++){

		if(validIntervals(dim,i) > -1) {

			validIntervalsForADimension(count) = i;
			count++;
		}

	}

	return validIntervalsForADimension;
}

uvec LHSSamples::returnAValidInterval(mat validIntervals){

	uvec intervals(numberOfDesignVariables);

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		uvec validIndices = returnValidIntervalsForADimension(validIntervals, i);
#if 0
		trans(validIndices).print();
#endif

#if 0
		cout<<"calling randomInterval\n";
#endif
		int randomInterval = generateRandomInt(validIndices);
		intervals(i)= randomInterval;


	}

	return intervals;

}




void LHSSamples::generateSamples(void){

#if 0
	cout<<"Generating "<<numberOfSamples<<" samples using LHS...\n";
#endif
	vec dx(numberOfDesignVariables);
	mat validIntervals(numberOfDesignVariables,numberOfSamples, fill::zeros);

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		dx(i) = (upperBounds(i) - lowerBounds(i))/numberOfSamples;

	}
#if 0
	printVector(upperBounds,"upperBounds");
	printVector(lowerBounds,"lowerBounds");
	printVector(dx,"dx");
#endif

	unsigned int numberOfSamplesGenerated = 0;

	uvec intervals(numberOfDesignVariables);
	vec lb(numberOfDesignVariables);
	vec ub(numberOfDesignVariables);


	while(numberOfSamplesGenerated < numberOfSamples){


		intervals = returnAValidInterval(validIntervals);
#if 0
		intervals.print();
#endif
		for(unsigned int i=0; i<numberOfDesignVariables; i++) {

			lb(i) = lowerBounds(i)+intervals(i)*dx(i)+dx(i)*0.45;
			ub(i) = lb(i) + 0.1*dx(i);
		}

		rowvec dv = generateRandomRowVector(lb,ub);
#if 0
		printVector(lb,"lb");
		printVector(ub,"ub");
		printVector(dv,"dv");
#endif


		for(unsigned int i=0; i<numberOfDesignVariables; i++) {

			validIntervals(i,intervals(i)) = -1;

		}
#if 0
		validIntervals.print();
#endif
		samples.row(numberOfSamplesGenerated) = dv;
		numberOfSamplesGenerated++;
#if 0
		cout<<"numberOfSamplesGenerated = "<<numberOfSamplesGenerated<<"\n";
#endif




	} /* end of while */


	if(testIfSamplesAreTooClose() == true){

		std::cout<<"WARNING: Samples are too close each other\n";

	}





}



void LHSSamples::roundSamplesToDiscreteValues(void){

	unsigned int howManyVariablesAreDiscrete = indicesDiscreteVariables.size();

	printVector(indicesDiscreteVariables);

	if(howManyVariablesAreDiscrete != 0){

		for(unsigned int i=0; i < samples.n_rows; i++){


			for(unsigned int j=0; j < howManyVariablesAreDiscrete; j++){


				unsigned int index = indicesDiscreteVariables[j];
				double valueToRound = samples(i,index);

				double dx = incrementsDiscreteVariables[j];
				unsigned int howManyDiscreteValues = (upperBounds(index) - lowerBounds(index))/dx;
				howManyDiscreteValues += 1;

				vec discreteValues(howManyDiscreteValues);

				discreteValues(0) = lowerBounds(index);
				for(unsigned int k=1; k<howManyDiscreteValues-1; k++){

					discreteValues(k) = discreteValues(k-1) + incrementsDiscreteVariables[j];


				}

				discreteValues(howManyDiscreteValues-1) = upperBounds(index);

				int whichInterval = findInterval(valueToRound, discreteValues);


				assert(whichInterval>=0);

				double distance1 = valueToRound - discreteValues[whichInterval];
				double distance2 = discreteValues[whichInterval+1] - valueToRound;

				if(distance1 < distance2){

					samples(i,index) =  discreteValues[whichInterval];

				}
				else{

					samples(i,index) =  discreteValues[whichInterval+1];
				}

				samples.print();

		}


	}



}

}



bool LHSSamples::testIfSamplesAreTooClose(void){

	bool ifTooClose = false;
#if 0
	std::cout<<"Testing Latin Hypercube samples...\n";
#endif

	if(samples.n_rows == 0){

		std::cout<<"ERROR: There are no samples to check!\n";
		abort();

	}

	vec maximumDx(numberOfDesignVariables);

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		maximumDx(i) = 0.5*(upperBounds(i) - lowerBounds(i))/numberOfSamples;

	}


	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec sample1 = samples.row(i);
		for(unsigned int j=0; j<numberOfSamples; j++){

			rowvec sample2 = samples.row(j);
			for(unsigned int k=0; k<this->numberOfDesignVariables; k++){

				if(i!=j){

					double dx = fabs(sample1(k) - sample2(k));
					if(dx < maximumDx(k)){
						sample1.print();
						sample2.print();
						std::cout<<"dx = "<<dx<<"\n";
						std::cout<<"maximumDx(k) = "<<maximumDx(k)<<"\n";
						ifTooClose = true;
						return ifTooClose;
					}

				}

			}

		}

	}




	return ifTooClose;

}



void LHSSamples::saveSamplesToCSVFile(std::string fileName){


	saveMatToCVSFile(samples,fileName);

}

void LHSSamples::visualize(void){

	if(this->numberOfDesignVariables!=2){
		cout<<"ERROR: Can only visualize 2D samples\n";
		abort();

	}
	saveSamplesToCSVFile("lhs_visialization.csv");
	std::string python_command = "python -W ignore "+ settings.python_dir + "/lhs.py lhs_visialization.dat";

#if 0
	cout<<python_command<<"\n";
#endif
	FILE* in = popen(python_command.c_str(), "r");

	fprintf(in, "\n");

}


void LHSSamples::printSamples(void){


	printMatrix(this->samples,"LHS Samples");



}


mat LHSSamples::getSamples(void){

	return this->samples;

}





RandomSamples::RandomSamples(unsigned int d, double lb, double ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	assert(ub>lb);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);
	upperBounds.fill(ub);
	lowerBounds.fill(lb);

	samples = zeros<mat>(N,d);
	generateSamples();

}

RandomSamples::RandomSamples(unsigned int d, double *lb, double *ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	for(unsigned int i=0; i<d; i++){

		assert(ub[i]>lb[i]);
	}


	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);

	for(unsigned int i=0; i<d; i++){
		upperBounds(i) = ub[i];
		lowerBounds(i) = lb[i];
	}

	samples = zeros<mat>(N,d);
	generateSamples();

}

RandomSamples::RandomSamples(unsigned int d, vec lb, vec ub, unsigned int N){

	assert(d>0);
	assert(N>d);
	numberOfDesignVariables = d;
	numberOfSamples = N;
	for(unsigned int i=0; i<d; i++){

		assert(ub(i)>lb(i));
	}


	upperBounds= ub;
	lowerBounds= lb;

	samples = zeros<mat>(N,d);

	generateSamples();


}

void RandomSamples::generateSamples(void){

	cout<<"Generating "<<numberOfSamples<<" samples using LHS...\n";

	vec dx(numberOfDesignVariables);
	mat validIntervals(numberOfDesignVariables,numberOfSamples, fill::zeros);

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		dx(i) = (upperBounds(i) - lowerBounds(i))/numberOfSamples;

	}
#if 1
	printVector(dx,"dx");
#endif

	unsigned int numberOfSamplesGenerated = 0;

	while(numberOfSamplesGenerated < numberOfSamples){


		rowvec dv = generateRandomRowVector(lowerBounds,upperBounds);
#if 1
		printVector(dv,"dv");
#endif

		samples.row(numberOfSamplesGenerated) = dv;
		numberOfSamplesGenerated++;

	} /* end of while */


}

void RandomSamples::saveSamplesToCSVFile(std::string fileName){


	saveMatToCVSFile(samples,fileName);

}

void RandomSamples::visualize(void){

	if(numberOfDesignVariables!=2){
		cout<<"ERROR: Can only visulaize 2D samples\n";
		abort();

	}
	saveSamplesToCSVFile("random_samples_visialization.csv");
	std::string python_command = "python -W ignore "+ settings.python_dir + "/lhs.py random_samples_visialization.dat";

#if 1
	cout<<python_command<<"\n";
#endif
	FILE* in = popen(python_command.c_str(), "r");

	fprintf(in, "\n");

}


void RandomSamples::printSamples(void){


	printMatrix(this->samples,"Random Samples");



}

void testRandomSamples2D(void){
	cout<<__func__<<"\n";
	RandomSamples DoE(2,0.0,1.0, 50);
	DoE.visualize();


}





FullFactorialSamples::FullFactorialSamples(unsigned int d, double lb, double ub, unsigned int levels){

	assert(d>0);
	assert(ub>lb);
	numberOfDesignVariables = d;
	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);

	numberOfLevels = zeros<uvec>(d);
	numberOfLevels.fill(levels);

	upperBounds.fill(ub);
	lowerBounds.fill(lb);


	numberOfSamples = pow(levels,d);

	cout<<"Number of Samples = "<<numberOfSamples<<"\n";

	samples = zeros<mat>(numberOfSamples,d);


	generateSamples();

}

FullFactorialSamples::FullFactorialSamples(unsigned int d, double *lb, double *ub, unsigned int levels){

	assert(d>0);
	numberOfDesignVariables = d;

	for(unsigned int i=0; i<d; i++){

		assert(ub[i]>lb[i]);
	}


	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);

	for(unsigned int i=0; i<d; i++){
		upperBounds(i) = ub[i];
		lowerBounds(i) = lb[i];
	}

	numberOfLevels = zeros<uvec>(d);
	numberOfLevels.fill(levels);

	numberOfSamples = pow(levels,d);


	samples = zeros<mat>(numberOfSamples,d);
	generateSamples();

}

FullFactorialSamples::FullFactorialSamples(unsigned int d, vec lb, vec ub, unsigned int levels){

	assert(d>0);

	numberOfDesignVariables = d;
	numberOfSamples = pow(levels,d);
	for(unsigned int i=0; i<d; i++){

		assert(ub(i)>lb(i));
	}


	upperBounds= ub;
	lowerBounds= lb;

	numberOfLevels = zeros<uvec>(d);
	numberOfLevels.fill(levels);

	samples = zeros<mat>(numberOfSamples,d);

	generateSamples();


}
FullFactorialSamples::FullFactorialSamples(unsigned int d, double lb, double ub, uvec levels){

	assert(d>0);
	assert(ub>lb);

	assert(levels.size() == d);

	numberOfDesignVariables = d;
	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);
	upperBounds.fill(ub);
	lowerBounds.fill(lb);

	numberOfLevels = levels;

	int N=1;

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		N = N * levels(i);
	}



	samples = zeros<mat>(N,d);

	numberOfSamples = N;

	generateSamples();

}

FullFactorialSamples::FullFactorialSamples(unsigned int d, double *lb, double *ub, uvec levels){

	assert(d>0);
	assert(levels.size() == d);
	numberOfDesignVariables = d;

	for(unsigned int i=0; i<d; i++){

		assert(ub[i]>lb[i]);
	}


	upperBounds= zeros<vec>(d);
	lowerBounds= zeros<vec>(d);

	for(unsigned int i=0; i<d; i++){
		upperBounds(i) = ub[i];
		lowerBounds(i) = lb[i];
	}
	numberOfLevels = levels;

	int N=1;

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		N = N * levels(i);
	}




	samples = zeros<mat>(N,d);
	generateSamples();

}

FullFactorialSamples::FullFactorialSamples(unsigned int d, vec lb, vec ub, uvec levels){

	assert(d>0);
	assert(levels.size() == d);
	numberOfDesignVariables = d;

	for(unsigned int i=0; i<d; i++){

		assert(ub(i)>lb(i));
	}

	upperBounds= ub;
	lowerBounds= lb;

	numberOfLevels = levels;

	int N=1;

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		N = N * levels(i);
	}

	numberOfSamples = N;


	samples = zeros<mat>(N,d);

	generateSamples();


}

void FullFactorialSamples::incrementIndexCount(uvec &indxCount){

	indxCount(0)++;

	for(unsigned int i=1; i<numberOfDesignVariables; i++){

		if(indxCount(i-1) == numberOfLevels(i-1)){

			indxCount(i)++;
			indxCount(i-1) = 0;
		}
	}
#if 1
	trans(indxCount).print();
#endif

}



void FullFactorialSamples::generateSamples(void){


	unsigned int numberOfSamplesGenerated = 0;

	uvec indxCount(numberOfDesignVariables);
	indxCount.fill(0);

	vec dx(numberOfDesignVariables);

	for(unsigned int i=0; i<numberOfDesignVariables; i++){

		dx(i) = (upperBounds(i) - lowerBounds(i))/numberOfLevels(i);


	}

#if 1
	printVector(dx,"dx");
#endif

	while(numberOfSamplesGenerated< numberOfSamples){

		rowvec dv(numberOfDesignVariables);


		for(unsigned int i=0; i<numberOfDesignVariables; i++){

			dv(i) = lowerBounds(i)+ 0.5*dx(i) + dx(i)*indxCount(i);

		}

#if 1
		printVector(dv,"dv");
#endif

		incrementIndexCount(indxCount);
		samples.row(numberOfSamplesGenerated) = dv;
		numberOfSamplesGenerated++;

	}


}

void FullFactorialSamples::saveSamplesToCSVFile(std::string fileName){

	saveMatToCVSFile(samples,fileName);

}

void FullFactorialSamples::visualize(void){

	if(this->numberOfDesignVariables!=2){
		cout<<"ERROR: Can only visulaize 2D samples\n";
		abort();

	}
	saveSamplesToCSVFile("full_factorial_samples_visialization.csv");
	std::string python_command = "python -W ignore "+ settings.python_dir + "/lhs.py full_factorial_samples_visialization.dat";

#if 1
	cout<<python_command<<"\n";
#endif
	FILE* in = popen(python_command.c_str(), "r");

	fprintf(in, "\n");

}

void FullFactorialSamples::printSamples(void){


	printMatrix(this->samples,"Full Factorial Samples");



}

void testFullFactorial2D(void){
	cout<<__func__<<"\n";
	uvec nLevels(2);
	nLevels(0) = 6;
	nLevels(1) = 6;

	FullFactorialSamples DoE(2,0.0,1.0, nLevels);
	DoE.visualize();


}
