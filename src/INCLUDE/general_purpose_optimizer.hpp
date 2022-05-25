/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 *  file is part of RoDeO
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
#ifndef GP_OPTIMIZER_HPP
#define GP_OPTIMIZER_HPP


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include "bounds.hpp"
#include "output.hpp"

using namespace arma;
using namespace std;

typedef double (*GeneralPurposeObjectiveFunction)(vec);

class GeneralPurposeOptimizer{

protected:

	unsigned int dimension = 0;
	Bounds parameterBounds;

	OutputDevice output;

	std::string problemName;
	std::string filenameOptimizationHistory;
	std::string filenameWarmStart;
	std::string filenameOptimizationResult;

	unsigned int maxNumberOfFunctionEvaluations = 0.0;

	double (*calculateObjectiveFunction)(vec);

	bool ifObjectiveFunctionIsSet = false;
	bool ifProblemNameIsSet = false;
	bool ifFilenameOptimizationHistoryIsSet = false;
	bool ifFilenameWarmStartIsSet = false;
	bool ifFilenameOptimizationResultIsSet = false;



	unsigned int numberOfThreads = 1;



public:

	void setDimension(unsigned int dim);
	unsigned int getDimension(void) const;


	void setBounds(double, double);
	void setBounds(vec, vec);
	void setBounds(Bounds);

	bool areBoundsSet(void) const;

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setProblemName(std::string);
	void setFilenameOptimizationHistory(std::string);
	void setFilenameWarmStart(std::string);
	void setFilenameOptimizationResult(std::string);


	bool isProblemNameSet(void);
	bool isFilenameOptimizationHistorySet(void);
	bool isFilenameWarmStartSet(void);
	bool isFilenameOptimizationResultSet(void);





	void setNumberOfThreads(unsigned int nTreads);
	unsigned int getNumberOfThreads(void) const;

	void setObjectiveFunction(GeneralPurposeObjectiveFunction );
	bool isObjectiveFunctionSet(void) const;

	void setMaxNumberOfFunctionEvaluations(unsigned int);

	virtual void optimize(void);
	virtual double calculateObjectiveFunctionInternal(vec &);

	double callObjectiveFunction(vec &);

};



#endif
