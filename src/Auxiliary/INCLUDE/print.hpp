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

#ifndef PRINT_HPP
#define PRINT_HPP

#include <armadillo>
#include <vector>
#include <cassert>
using namespace arma;
using std::string;


#define PRINT_HERE printf("Here : %s(%d)--:\n",__FILE__,__LINE__);

#define printScalar(name) printScalarValueWithName(#name, (name))
#define printTwoScalars(name1,name2) printTwoScalarValuesWithNames(#name1, (name1), #name2, (name2))

void printScalarValueWithName(std::string name, int value) ;
void printScalarValueWithName(std::string name, double value);
void printScalarValueWithName(std::string name, unsigned int value);


void printTwoScalarValuesWithNames(std::string name1, double value1,std::string name2, double value2 );



void printMessage(std::string msg);

void printList(std::vector<int> list, std::string msg);
void printList(std::vector<unsigned int> list, std::string msg);
void printList(std::vector<int> list);
void printList(std::vector<unsigned int> list);


#endif
