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

#include "metric.hpp"
#include<gtest/gtest.h>

TEST(testMetric, testfindNearestNeighborL1){

	mat X(100,5, fill::randu);

	rowvec xp(5,fill::randu);
	xp+=0.000001;

	X.row(11) = xp;
	unsigned int indx = findNearestNeighborL1(xp, X);

	ASSERT_EQ(indx, 11);

}

TEST(testMetric, testcalculateL1norm){

	rowvec xp(4); xp(0) = -1; xp(1) = 1; xp(2) = -2; xp(3) = 2;

	double normL1 = calculateL1norm(xp);
	double error = fabs(normL1-6.0);
	EXPECT_LT(error, 10E-10);



}




