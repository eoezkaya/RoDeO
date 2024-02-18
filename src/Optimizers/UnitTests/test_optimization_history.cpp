/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), Rheinland-Pfälzische Technische Universität
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

#include "../INCLUDE/optimization_history.hpp"
#include<gtest/gtest.h>

class OptimizationHistoryTest: public ::testing::Test {
protected:
	void SetUp() override {

		testHistory.addConstraintName("area");
		testHistory.addConstraintName("volume");
		testHistory.addConstraintName("stress");

		testHistory.setDimension(3);
		testHistory.setObjectiveFunctionName("drag");

		mat somedata(20,9,fill::randu);

		testHistory.setData(somedata);

	}

	void TearDown() override {

	}

	OptimizationHistory testHistory;

};

TEST_F(OptimizationHistoryTest, constructor){

	OptimizationHistory testObject;
	ASSERT_TRUE(testObject.dimension == 0);


}

TEST_F(OptimizationHistoryTest, setHeader){

	field<std::string> header = testHistory.setHeader();
	ASSERT_TRUE(header.size() > 0);
}

TEST_F(OptimizationHistoryTest, saveOptimizationHistoryFile){

	testHistory.saveOptimizationHistoryFile();

}
TEST_F(OptimizationHistoryTest, updateOptimizationHistory){

	Design d(3);
	rowvec dv(3,fill::randu);
	double J = 3.22;
	d.designParameters = dv;
	rowvec constraints(3, fill::randu);
	d.constraintTrueValues = constraints;
	d.improvementValue = 4.44;
	d.trueValue = J;

	testHistory.updateOptimizationHistory(d);

	mat updatedData = testHistory.getData();
	ASSERT_TRUE(updatedData.n_rows == 21);

}

TEST_F(OptimizationHistoryTest, calculateInitialImprovementValue){

	mat data(5,9,fill::randu);
	data.col(8).fill(0.0);
	data(4,8) = 1.0;
	data(0,3) = 1;
	data(1,3) = 2;
	data(2,3) = 3;
	data(3,3) = 4;
	data(4,3) = 3;

//	data.print();
	testHistory.setData(data);
	double val = testHistory.calculateInitialImprovementValue();

	ASSERT_TRUE(fabs(val-3) < 10E-10);

}
