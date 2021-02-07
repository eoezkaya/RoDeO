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
#ifndef DRIVERS_HPP
#define DRIVERS_HPP


#include<armadillo>
#include<vector>
using namespace arma;




class RoDeODriver{

private:

	std::string *keywords;
	unsigned int numberOfKeywords;

	std::string configFileName;
	std::string problemType;
	std::string problemName;
	int dimension;
	int numberOfConstraints;
	vec boxConstraintsUpperBounds;
	vec boxConstraintsLowerBounds;
	vec constraintValues;

	bool ifObjectiveFunctionNameIsSet;
	bool ifObjectiveFunctionOutputFileIsSet;
	bool ifNumberOfConstraintsSet;
	bool ifProblemDimensionSet;
	bool ifUpperBoundsSet;
	bool ifLowerBoundsSet;
	bool ifProblemTypeSet;


	bool IsGradientsAvailableForObjectiveFunction;

	std::string objectiveFunctionName;
	std::vector<std::string> constraintNames;
	std::vector<std::string> constraintTypes;
	std::vector<std::string> executableNames;
	std::vector<std::string> executablePaths;
	std::string executableInputFile;
	std::vector<std::string> executableOutputFiles;


public:
	RoDeODriver();
	void checkConsistencyOfConfigParams(void);
	void readConfigFile(void);
	void runDriver(void);

};



#endif
