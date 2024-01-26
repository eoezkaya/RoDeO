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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include<string>
#include "../../LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"

using std::string;

void printRoDeOIntro(void);


class OutputDevice{


public:

	bool ifScreenDisplay = false;

	OutputDevice();
	void setDisplayOn(void);

	void printMessage(string) const;
	void printErrorMessageAndAbort(string message) const;

	void printMessage(string, string) const;
	void printMessage(string, int) const;
	void printMessage(string, unsigned int) const;
	void printMessage(string, double) const;
	void printMessage(string, double,string message2, double ) const;
	void printMessage(std::string, vec) const;
	void printMessage(std::string message, rowvec whatToPrint) const;
	void printMessage(std::string message, mat whatToPrint) const;
	void printDesign(const Design &d) const;
	void printDesign(const DesignForBayesianOptimization &d) const;
	void printIteration(unsigned int iteration) const;
	void printBoxConstraints(const vec &lb, const vec &ub) const;
	void printList(std::vector<int> list, std::string msg) const;


};

#endif

