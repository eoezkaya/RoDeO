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


#include <cassert>
#include "design.hpp"
#include "matrix_vector_operations.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


Design::Design(rowvec dv, unsigned int numberOfConstraints){

	unsigned int dimension = dv.size();
	designParameters = dv;
	constraintTrueValues = zeros<rowvec>(numberOfConstraints);
	gradient = zeros<rowvec>(dimension);


}

Design::Design(rowvec dv){

	unsigned int dimension = dv.size();
	designParameters = dv;
	gradient = zeros<rowvec>(dimension);


}

void Design::print(void) const{

	printVector(designParameters,"designParameters");
	std::cout<<"Value = "<<trueValue<<"\n";
	printVector(gradient,"gradient vector");
	printVector(constraintTrueValues,"constraint values");



	for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

		printVector(*it);


	}

}
