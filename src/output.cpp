/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#include "output.hpp"
#include <iostream>

using std::cout;

OutputDevice::OutputDevice(){


}

void OutputDevice::printMessage(string message) const{

	if(ifScreenDisplay){

		cout<<message<<"\n";

	}

}

void OutputDevice::printErrorMessageAndAbort(string message) const{

		string errorMessage ="ERROR: " + message;
		cout<<errorMessage<<"\n";
		abort();


}

void OutputDevice::printMessage(string message, string whatToPrint) const{

	if(ifScreenDisplay){

		cout<<message<<" "<<whatToPrint<<"\n";

	}

}


void OutputDevice::printMessage(std::string message, int whatToPrint) const{


	if(ifScreenDisplay){

		std::cout<<message<<" ";
		std::cout<<whatToPrint<<"\n";

	}


}

void OutputDevice::printMessage(std::string message, unsigned int whatToPrint) const{


	if(ifScreenDisplay){

		std::cout<<message<<" ";
		std::cout<<whatToPrint<<"\n";

	}


}


void OutputDevice::printMessage(std::string message, double whatToPrint) const{


	if(ifScreenDisplay){

		std::cout<<message<<" ";
		std::cout<<whatToPrint<<"\n";

	}


}

void OutputDevice::printMessage(std::string message, vec whatToPrint) const{


	if(ifScreenDisplay){

		printVector(whatToPrint,message);

	}


}

void OutputDevice::printMessage(std::string message, mat whatToPrint) const{


	if(ifScreenDisplay){

		printMatrix(whatToPrint,message);

	}


}
