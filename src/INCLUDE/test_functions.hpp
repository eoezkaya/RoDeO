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
#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "design.hpp"

#include <string>
#include <armadillo>



using namespace arma;


class TestFunction {

private:

	unsigned int dimension;
	std::string function_name;

	vec lb; /*lower bounds */
	vec ub; /*upper bounds */

	mat trainingSamples;
	mat testSamples;

	mat trainingSamplesInput;
	mat testSamplesInput;


	unsigned int numberOfTrainingSamples = 0;
	unsigned int numberOfTestSamples = 0;

	std::string filenameTrainingData;
	std::string filenameTestData;


	bool ifFunctionIsNoisy = false;
	bool ifBoxConstraintsSet = false;
	bool ifGradientsAvailable = false;

	bool ifFunctionPointerIsSet = false;

	bool ifVisualize = false;
	bool ifDisplayResults = false;

	bool ifTrainingDataFileExists = false;
	bool ifTestDataFileExists = false;

	double noiseLevel = 0.0;

	std::string fileNameSurrogateModelData;

	double (*func_ptr)(double *);
	double (*adj_ptr)(double *, double *);



public:

	double inSampleError = 0.0;
	double outSampleError = 0.0; /*generalization error */

	void evaluate(Design &d) const;
	void evaluateAdjoint(Design &d) const;

	short int numberOfSamplesUsedForVisualization = 100;



    TestFunction(std::string name,int dimension);

    void plot(int resolution = 100) const;

    void generateSamplesInputTrainingData(void);
    void generateSamplesInputTestData(void);

    void generateTestSamples(void);
    void generateTrainingSamples(void);

    mat getTrainingSamplesInput(void) const;
    mat getTestSamplesInput(void) const;
    mat getTrainingSamples(void) const;
    mat getTestSamples(void) const;



    void validateAdjoints(void);


    void setNoiseLevel(double);
    void setVisualizationOn(void);
    void setVisualizationOff(void);
    void setGradientsOn(void);
    void setGradientsOff(void);

    void setDisplayOn(void);
    void setDisplayOff(void);

    void setNumberOfTrainingSamples(unsigned int);
    void setNumberOfTestSamples(unsigned int);

    void setNameFilenameTrainingData(std::string);
    void setNameFilenameTestData(std::string );

    void readFileTestData(void);
    void readFileTrainingData(void);

    void visualizeSurrogate1D(SurrogateModel *TestFunSurrogate, unsigned int resolution=1000) const;
    void visualizeSurrogate2D(SurrogateModel *TestFunSurrogate, unsigned int resolution=100) const;

    void print(void);
    mat  generateRandomSamples(unsigned int);
    mat  generateRandomSamplesWithGradients(unsigned int);

    mat  generateUniformSamples(unsigned int howManySamples) const;

    void evaluateGlobalExtrema(void) const;
    void setBoxConstraints(double lowerBound, double upperBound);
    void setBoxConstraints(vec lowerBound, vec upperBound);


    void setFunctionPointer(double (*testFunction)(double *));
    void setFunctionPointer(double (*testFunctionAdjoint)(double *, double *));

    bool checkIfExecutableIsReadyToRun(void) const;

} ;







/* regression test functions */

double LinearTF1(double *x);
double LinearTF1Adj(double *x, double *xb);

void generateEggholderData(std::string filename, unsigned int nSamples);
void generateEggholderDataMultiFidelity(std::string filenameHiFi, std::string filenameLowFi, unsigned int nSamplesHiFi, unsigned int nSamplesLowFi);
double Eggholder(double *x);
double EggholderAdj(double *x, double *xb);



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

double Himmelblau(double *x);
double HimmelblauAdj(double *x, double *xb);
void generateHimmelblauDataMultiFidelity(std::string, std::string, unsigned int, unsigned int);
void generateHimmelblauDataMultiFidelityWithGradients(std::string, std::string, unsigned int, unsigned int);


double Borehole(double *x);
double BoreholeAdj(double *x, double *xb);


double Wingweight(double *x);
double WingweightAdj(double *xin, double *xb);




#endif
