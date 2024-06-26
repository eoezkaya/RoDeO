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


#include <armadillo>
#include <cassert>
#include <iostream>


#include "./INCLUDE/vector_operations.hpp"


void appendRowVectorToCSVData(rowvec v, std::string filename){

	assert(v.size() > 0);
	assert(isNotEmpty(filename));

	std::ofstream outfile;

	outfile.open(filename, std::ios_base::app); // append instead of overwrite

	outfile.precision(10);
	for(unsigned int i=0; i<v.size()-1; i++){

		outfile << v(i) <<",";
	}
	outfile << v(v.size()-1)<<"\n";
	outfile.close();

}







