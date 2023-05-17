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
#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "design.hpp"
#include "bounds.hpp"

#include <string>
#include <armadillo>



using namespace arma;


class TestFunction {

public:

	unsigned int dimension;
	std::string function_name;

	Bounds boxConstraints;

	mat trainingSamples;
	mat trainingSamplesLowFidelity;
	mat testSamples;

	mat trainingSamplesInput;
	mat testSamplesInput;


	unsigned int numberOfTrainingSamples = 0;
	unsigned int numberOfTrainingSamplesLowFi = 0;
	unsigned int numberOfTestSamples = 0;

	std::string filenameTrainingData;
	std::string filenameTestData;

	std::string filenameTrainingDataLowFidelity;
	std::string filenameTrainingDataHighFidelity;

	bool ifInputSamplesAreGenerated = false;
	bool ifSomeAdjointsAreLeftBlank = false;


	double noiseLevel = 0.0;


	double (*func_ptr)(double *) = NULL;
	double (*adj_ptr)(double *, double *) = NULL;
	double (*tan_ptr)(double *, double *, double *) = NULL;

	double (*func_ptrLowFi)(double *) = NULL;
	double (*adj_ptrLowFi)(double *, double *) = NULL;
	double (*tan_ptrLowFi)(double *, double *, double *) = NULL;

	unsigned int evaluationSelect = 1;

	void evaluate(Design &d) const;

	TestFunction(std::string name,int dimension);

    void setBoxConstraints(double lb, double ub);
    void setBoxConstraints(Bounds);


    void generateSamplesInputTrainingData(void);

    void generateSamplesInputTestData(void);
    void generateSamplesInputTestDataCloseToTrainingSamples(void);

    void generateTestSamples(void);
    void generateTestSamplesCloseToTrainingSamples(void);

    void generateTrainingSamples(void);
    void generateTrainingSamplesWithAdjoints(void);
    void generateTrainingSamplesWithTangents(void);

    void generateTrainingSamplesMultiFidelity(void);
    void generateTrainingSamplesMultiFidelityWithAdjoint(void);
    void generateTrainingSamplesMultiFidelityWithTangents(void);
    void generateTrainingSamplesMultiFidelityWithLowFiAdjoint(void);
    void generateTrainingSamplesMultiFidelityWithLowFiTangents(void);

    void print(void);
    void evaluateGlobalExtrema(void) const;

private:
    mat generateSamplesWithFunctionalValues(mat input, unsigned int N) const;
    mat generateSamplesWithAdjoints(mat input, unsigned int N) const;
    mat generateSamplesWithTangents(mat input, unsigned int N) const;
} ;







/* regression test functions */

/*****************************************************************************/

double LinearTF1(double *x);
double LinearTF1LowFidelity(double *x);
double LinearTF1LowFidelityAdj(double *x, double *xb);
double LinearTF1Adj(double *x, double *xb);
double LinearTF1Tangent(double *x, double *xd, double *fdot);
double LinearTF1LowFidelityTangent(double *x, double *xd, double *fdot);


/*****************************************************************************/
double testFunction1D(double *x);
double testFunction1DAdj(double *x, double *xb);
double testFunction1DTangent(double *x, double *xd, double *testFunction1D);
double testFunction1DLowFi(double *x);
double testFunction1DAdjLowFi(double *x, double *xb);
double testFunction1DTangentLowFi(double *x, double *xd, double *fdot);

/*****************************************************************************/

double Himmelblau(double *x);
double HimmelblauLowFi(double *x);
double HimmelblauAdj(double *x, double *xb);
double HimmelblauAdjLowFi(double *x, double *xb);
double HimmelblauTangent(double *x, double *xd, double *fdot);
double HimmelblauTangentLowFi(double *x, double *xd, double *fdot);

double himmelblauConstraintFunction1(double *x);
double himmelblauConstraintFunction2(double *x);


/*****************************************************************************/




void generateEggholderData(std::string filename, unsigned int nSamples);
void generateEggholderDataMultiFidelity(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi);


double Eggholder(double *x);
double EggholderAdj(double *x, double *xb);
double Eggholder(vec x);


double Rastrigin6D(double *x);
double Rastrigin6DAdj(double *xin, double *xb);



double Waves2D(double *x);
double Waves2DAdj(double *x,double *xb);

double Waves2DWithHighFrequencyPart(double *x);
double Waves2DWithHighFrequencyPartAdj(double *x,double *xb);

double Herbie2D(double *x);
double Herbie2DAdj(double *x, double *xb);


double McCormick(double *x);
double McCormickAdj(double *x, double *xb);

double GoldsteinPrice(double *x);
double GoldsteinPriceAdj(double *x, double *xb);

double Rosenbrock(double *x);
double RosenbrockAdj(double *x, double *xb);


double Rosenbrock3D(double *x);
double Rosenbrock3DAdj(double *x, double *xb);

double Rosenbrock4D(double *x);
double Rosenbrock4DAdj(double *x, double *xb);



double Rosenbrock8D(double *x);
double Rosenbrock8DAdj(double *x, double *xb) ;

double Shubert(double *x);
double ShubertAdj(double *x, double *xb);


double Himmelblau(vec x);
double HimmelblauLowFi(vec x);
vec HimmelblauGradient(vec x);

void generateHimmelblauDataMultiFidelity(std::string, std::string, unsigned int, unsigned int);
void generateHimmelblauDataMultiFidelityWithShuffle(std::string , std::string, unsigned int, unsigned int);

void generateHimmelblauDataMultiFidelityWithGradients(std::string, std::string, unsigned int, unsigned int);


double Borehole(double *x);
double BoreholeAdj(double *x, double *xb);


double Wingweight(double *x);
double WingweightAdj(double *xin, double *xb);


double Alpine02_5D(double *x);
double Alpine02_5DAdj(double *x, double *xb);
double Alpine02_5DTangent(double *x, double *xd, double *fdot);






#endif
