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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#ifndef LHSSAMPLING
#define LHSSAMPLING
#include <string>
#include <armadillo>
#include "../../INCLUDE/Rodeo_globals.hpp"
using namespace arma;



class LHSSamples{

private:
	unsigned int numberOfDesignVariables;
	unsigned int numberOfSamples;

	vec lowerBounds;
	vec upperBounds;

	std::vector<int> indicesDiscreteVariables;
	vec incrementsDiscreteVariables;


	mat samples;

	void generateSamples(void);

	uvec returnAValidInterval(mat validIntervals);
	uvec returnValidIntervalsForADimension(mat validIntervals, unsigned int dim);


public:

	LHSSamples(unsigned int d, double lb, double ub, unsigned int N);
	LHSSamples(unsigned int d, vec lb, vec ub, unsigned int N);
	LHSSamples(unsigned int d, double *lb, double* ub, unsigned int N);

	void setDiscreteParameterIndices(int *indices, int size);
	void setDiscreteParameterIncrements(vec increments);

	void setDiscreteParameterIndices(std::vector<int>);
	void setDiscreteParameterIncrements(std::vector<double>);


	void saveSamplesToCSVFile(std::string);
	void printSamples(void);
	mat getSamples(void);

	void roundSamplesToDiscreteValues(void);

	bool testIfSamplesAreTooClose(void);


	/* test functions */

	friend void testLHS2D(void);



};


class RandomSamples{

private:
	unsigned int numberOfDesignVariables;
	unsigned int numberOfSamples;
	vec lowerBounds;
	vec upperBounds;
	mat samples;

	void generateSamples(void);




public:

	RandomSamples(unsigned int d, double lb, double ub, unsigned int N);
	RandomSamples(unsigned int d, vec lb, vec ub, unsigned int N);
	RandomSamples(unsigned int d, double *lb, double* ub, unsigned int N);

	void saveSamplesToCSVFile(std::string filename);
	void printSamples(void);

	/* test functions */

	friend void testRandomSamples2D(void);



};

class FullFactorialSamples{

private:
	unsigned int numberOfDesignVariables;
	unsigned int numberOfSamples;
	uvec numberOfLevels;
	vec lowerBounds;
	vec upperBounds;
	mat samples;


	void generateSamples(void);
	void incrementIndexCount(uvec &indxCount);



public:

	FullFactorialSamples(unsigned int d, double lb, double ub, unsigned int levels);
	FullFactorialSamples(unsigned int d, vec lb, vec ub, unsigned int levels);
	FullFactorialSamples(unsigned int d, double *lb, double* ub, unsigned int levels);

	FullFactorialSamples(unsigned int d, double lb, double ub, uvec levels);
	FullFactorialSamples(unsigned int d, vec lb, vec ub, uvec levels);
	FullFactorialSamples(unsigned int d, double *lb, double* ub, uvec levels);


	void saveSamplesToCSVFile(std::string filename);
	void printSamples(void);


	/* test functions */

	friend void testFullFactorial2D(void);



};


#endif
