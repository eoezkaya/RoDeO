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

#include "trust_region_gek.hpp"
#include "aggregation_test.hpp"
#include "auxiliary_functions.hpp"
#include "test_functions.hpp"
#include "polynomials.hpp"
#include <vector>
#include <armadillo>


void testAllAggregationModel(void){

	testAggregationModeltrain();

}

void testAggregationModeltrain(void){
	cout<<__func__<<"\n";
	int dim = generateRandomInt(4,8);
	int N =  generateRandomInt(100,200);
	generateRandomTestAndValidationDataForGradientModels(dim,N);

	AggregationModel testModel("testData");


	testModel.train();

	cout<<__func__<<" is finished\n";

}

void testAggregationModelHimmelblau(void){


	TestFunction testFunHimmelblau("Himmelblau",2);
//	testFunHimmelblau.adj_ptr = HimmelblauAdj;
	testFunHimmelblau.func_ptr = Himmelblau;
	testFunHimmelblau.setVisualizationOn();
	testFunHimmelblau.setBoxConstraints(0.0,5.0);
	testFunHimmelblau.testSurrogateModel(KERNEL_REGRESSION,100);


}


void testAggregationInterpolate(void){
	cout<<__func__<<"\n";
	int dim = generateRandomInt(4,8);
	int N =  generateRandomInt(100,200);

	generateRandomTestAndValidationDataForGradientModels(dim,N);

	AggregationModel testModel("testData");


	testModel.train();

	mat validationData;

	validationData.load("testDataValidationPoints.csv",arma::csv_ascii);

	validationData.print("validationData");

	mat Xvalidation = validationData.submat(0,0,validationData.n_rows-1,dim-1);
	Xvalidation = (1.0/dim)*Xvalidation;


	vec yvalidation = validationData.col(dim);

	printMatrix(Xvalidation,"Xvalidation");
	printVector(yvalidation,"yvalidation");



	for(unsigned int i=0; i<validationData.n_rows; i++){

		rowvec x= Xvalidation.row(i);

		double fSurrogate = testModel.interpolate(x);
		cout<<"\nfSurrogate = "<<fSurrogate<<"\n";
		cout<<"fExact = "<<yvalidation(i)<<"\n\n";

	}



	cout<<__func__<<" is finished\n";

}




















