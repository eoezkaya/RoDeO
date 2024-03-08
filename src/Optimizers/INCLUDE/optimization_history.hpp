/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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
#ifndef OPT_HISTORY_HPP
#define OPT_HISTORY_HPP

#include <armadillo>
#include "./design.hpp"

#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif

using namespace std;
using namespace arma;
class OptimizationHistory{

#ifdef UNIT_TESTS
	friend class OptimizationHistoryTest;
	FRIEND_TEST(OptimizationHistoryTest, constructor);
	FRIEND_TEST(OptimizationHistoryTest, setHeader);
	FRIEND_TEST(OptimizationHistoryTest, calculateCrowdingFactor);
#endif

private:

	string filename = "optimizationHistory.csv";
	string objectiveFunctionName = "objective";
	vector<string> constraintNames;
	vector<string> variableNames;

	mat data;
	unsigned int dimension = 0;


	double crowdingFactor = 0;

public:

	unsigned int numberOfDoESamples = 0;

	void setFileName(string);
	void setDimension(unsigned int);
	void setData(mat);

	void reset(void);

	mat getData(void) const;
	vec getObjectiveFunctionValues(void) const;
	vec getFeasibilityValues(void) const;
	double getCrowdingFactor(void) const;

	field<std::string> setHeader(void) const;
	void addConstraintName(string);
	void setObjectiveFunctionName(string);

	void saveOptimizationHistoryFile(void);

	void updateOptimizationHistory(Design d);

	double calculateInitialImprovementValue(void) const;
	void print(void) const;

	void calculateCrowdingFactor(void);

};


#endif
