/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "standard_test_functions.hpp"
#include "bounds.hpp"
#include "random_functions.hpp"
#include "matrix_vector_operations.hpp"
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
	function.numberOfTrainingSamplesLowFi = 100;


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
	function.numberOfTrainingSamplesLowFi = 100;

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




