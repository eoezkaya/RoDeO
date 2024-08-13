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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "standard_test_functions.hpp"
#include "bounds.hpp"
#include "random_functions.hpp"
#include "vector_operations.hpp"
#include <cassert>

HimmelblauFunction::HimmelblauFunction():function("Himmelblau", 2){

	function.func_ptr = Himmelblau;
	function.adj_ptr  = HimmelblauAdj;
	function.tan_ptr = HimmelblauTangent;
	function.func_ptrLowFi = HimmelblauLowFi;
	function.adj_ptrLowFi = HimmelblauAdjLowFi;
	function.tan_ptrLowFi = HimmelblauTangentLowFi;

	function.setBoxConstraints(-6.0, 6.0);
	function.filenameTestData = "himmelblauTestData.csv";
	function.filenameTrainingData = "himmelblauTrainingData.csv";
	function.filenameTrainingDataLowFidelity = "himmelblauLowFiTrainingData.csv";
	function.filenameTrainingDataHighFidelity = function.filenameTrainingData;

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;
	function.numberOfTrainingSamplesLowFi = 0;


}

HimmelblauFunction::HimmelblauFunction(double lb, double ub):function("Himmelblau", 2){

	function.func_ptr = Himmelblau;
	function.adj_ptr  = HimmelblauAdj;
	function.tan_ptr = HimmelblauTangent;
	function.func_ptrLowFi = HimmelblauLowFi;
	function.adj_ptrLowFi = HimmelblauAdjLowFi;
	function.tan_ptrLowFi = HimmelblauTangentLowFi;

	function.filenameTestData = "himmelblauTestData.csv";
	function.filenameTrainingData = "himmelblauTrainingData.csv";
	function.filenameTrainingDataLowFidelity = "himmelblauLowFiTrainingData.csv";
	function.filenameTrainingDataHighFidelity = function.filenameTrainingData;

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;
	function.numberOfTrainingSamplesLowFi = 0;

	function.setBoxConstraints(lb, ub);

}


/***********************************************************************************/

LinearTestFunction1::LinearTestFunction1():function("LinearTestFunction1", 2){

	function.func_ptr = LinearTF1;
	function.adj_ptr  = LinearTF1Adj;
	function.tan_ptr  = LinearTF1Tangent;
	function.func_ptrLowFi = LinearTF1LowFidelity;
	function.adj_ptrLowFi  = LinearTF1LowFidelityAdj;
	function.tan_ptrLowFi  = LinearTF1LowFidelityTangent;
	function.setBoxConstraints(-3.0, 3.0);


}

LinearTestFunction1::LinearTestFunction1(double lb, double ub):function("LinearTestFunction1", 2){

	function.func_ptr = LinearTF1;
	function.adj_ptr  = LinearTF1Adj;
	function.tan_ptr  = LinearTF1Tangent;
	function.func_ptrLowFi = LinearTF1LowFidelity;
	function.adj_ptrLowFi  = LinearTF1LowFidelityAdj;
	function.tan_ptrLowFi  = LinearTF1LowFidelityTangent;
	function.setBoxConstraints(lb, ub);

}


/***********************************************************************************/









NonLinear1DTestFunction1::NonLinear1DTestFunction1():function("NonLinear1DTestFunction1", 1){

	function.func_ptr = testFunction1D;
	function.adj_ptr  = testFunction1DAdj;
	function.tan_ptr  = testFunction1DTangent;
	function.func_ptrLowFi = testFunction1DLowFi;
	function.adj_ptrLowFi  = testFunction1DAdjLowFi;
	function.tan_ptrLowFi  = testFunction1DTangentLowFi;
	function.setBoxConstraints(-3.0, 3.0);


}

NonLinear1DTestFunction1::NonLinear1DTestFunction1(double lb, double ub):function("NonLinear1DTestFunction1", 1){

	function.func_ptr = testFunction1D;
	function.adj_ptr  = testFunction1DAdj;
	function.tan_ptr  = testFunction1DTangent;
	function.func_ptrLowFi = testFunction1DLowFi;
	function.adj_ptrLowFi  = testFunction1DAdjLowFi;
	function.tan_ptrLowFi  = testFunction1DTangentLowFi;
	function.setBoxConstraints(lb, ub);

}

/***********************************************************************************/

HimmelblauConstraintFunction1::HimmelblauConstraintFunction1():function("ConstraintFunction1", 2){

	function.func_ptr = himmelblauConstraintFunction1;
	function.setBoxConstraints(-6.0, 6.0);

}


HimmelblauConstraintFunction2::HimmelblauConstraintFunction2():function("ConstraintFunction2", 2){

	function.func_ptr = himmelblauConstraintFunction2;
	function.setBoxConstraints(-6.0, 6.0);

}


WingweightFunction::WingweightFunction():function("Himmelblau", 10){

	function.func_ptr = Wingweight;
	function.adj_ptr  = WingweightAdj;

	Bounds boxConstraints;

	vec lb(10);
	vec ub(10);

	lb(0) = 150.0; ub(0) = 200.0;
	lb(1) = 220.0; ub(1) = 300.0;
	lb(2) = 6.0;   ub(2) = 10.0;
	lb(3) = -10.0; ub(3) = 10.0;
	lb(4) = 16.0;  ub(4) = 45.0;
	lb(5) = 0.5;   ub(5) = 1.0;
	lb(6) = 0.08;  ub(6) = 0.18;
	lb(7) = 2.5;   ub(7) = 6.0;
	lb(8) = 1700.0;ub(8) = 2500.0;
	lb(9) = 0.025; ub(9) = 0.08;


	boxConstraints.setBounds(lb,ub);
	function.setBoxConstraints(boxConstraints);
	function.filenameTestData = "wingweightTestData.csv";
	function.filenameTrainingData = "wingweight.csv";

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;

}


Alpine02_5DFunction::Alpine02_5DFunction():function("Alpine02_5D", 5){

	function.func_ptr = Alpine02_5D;
	function.adj_ptr  = Alpine02_5DAdj;
	function.tan_ptr  = Alpine02_5DTangent;
	Bounds boxConstraints;

	vec lb(5);
	vec ub(5);

	lb(0) = 0.0; ub(0) = 10.0;
	lb(1) = 0.0; ub(1) = 10.0;
	lb(2) = 0.0; ub(2) = 10.0;
	lb(3) = 0.0; ub(3) = 10.0;
	lb(4) = 0.0; ub(4) = 10.0;


	boxConstraints.setBounds(lb,ub);
	function.setBoxConstraints(boxConstraints);
	function.filenameTestData = "alpine02_5DTestData.csv";
	function.filenameTrainingData = "alpine02_5D.csv";

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;

}

EggholderFunction::EggholderFunction():function("Eggholder", 2){

	function.func_ptr = Eggholder;
	function.adj_ptr  = EggholderAdj;
	Bounds boxConstraints;

	vec lb(2);
	vec ub(2);

	lb(0) = 0.0; ub(0) = 512.0;
	lb(1) = 0.0; ub(1) = 512.0;

	boxConstraints.setBounds(lb,ub);
	function.setBoxConstraints(boxConstraints);
	function.filenameTestData = "eggholderTestData.csv";
	function.filenameTrainingData = "eggholder.csv";

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;

}

/***********************************************************************************************/

Griewank2DFunction::Griewank2DFunction():function("Griewank2D", 2){

	function.func_ptr  = griewank2D;
	function.adj_ptr   = griewank2DAdjoint;
	function.tan_ptr   = griewank2DTangent;
	Bounds boxConstraints;

	vec lb(2);
	vec ub(2);

	lb(0) = -600.0; ub(0) = 600.0;
	lb(1) = -600.0; ub(1) = 600.0;

	boxConstraints.setBounds(lb,ub);
	function.setBoxConstraints(boxConstraints);
	function.filenameTestData = "griewank2DTestData.csv";
	function.filenameTrainingData = "griewank2D.csv";

	function.numberOfTrainingSamples = 50;
	function.numberOfTestSamples = 100;

}
