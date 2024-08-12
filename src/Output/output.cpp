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

#include "output.hpp"
#include "design.hpp"

#include <iostream>

using std::cout;


void printIntro(void) {
	printf("\n\n\n");
	printf("	  ____       ____        ___   \n");
	printf("	 |  _ \\ ___ |  _ \\  ___ / _ \\  \n");
	printf("	 | |_) / _ \\| | | |/ _ \\ | | | \n");
	printf("	 |  _ < (_) | |_| |  __/ |_| | \n");
	printf("	 |_| \\_\\___/|____/ \\___|\\___/  \n");
	printf("\n");
	printf("    RObust DEsign Optimization Package      ");
	printf("\n\n\n");
}


std::string generateFormattedString(std::string msg, char c, int totalLength) {
    if (totalLength < 0) {
        throw std::invalid_argument("Number of characters must be non-negative.");
    }

    if(msg.length()%2 == 1){
    	msg+=" ";
    }

    int numEquals = (totalLength - msg.length() - 2)/2;


    std::string border(numEquals, c);



    std::ostringstream oss;
    oss << border << " " << msg << " " << border;

    return oss.str();
}


std::string generateFormattedString(std::string& content) {
    const int totalWidth = 120; // total width of the line
    const std::string border(totalWidth, '*'); // create a border line with '#'

    if (content.length() > totalWidth) {
        throw std::invalid_argument("Content length exceeds the total width.");
    }

    std::ostringstream oss;
    oss << border << "\n";

    int padding = (totalWidth - content.length()) / 2;
    int extraPadding = (totalWidth - content.length()) % 2; // handle odd lengths for perfect centering

    oss << std::string(padding, ' ') << content << std::string(padding + extraPadding, ' ') << "\n";
    oss << border;

    return oss.str();
}


std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Function to generate the formatted string
std::string generateProcessStartMessage() {
    std::string dateTime = getCurrentDateTime();
    std::ostringstream oss;

    std::string border(100, '=');
    oss << border <<"\n"
        << "Process start: \"" << dateTime << "\"\n"
        << border <<"\n";

    return oss.str();
}

std::string generateProcessEndMessage() {
    std::string dateTime = getCurrentDateTime();
    std::ostringstream oss;

    std::string border(100, '=');
    oss << border <<"\n"
           << "Process end: \"" << dateTime << "\"\n"
           << border <<"\n";

    return oss.str();
}


OutputDevice::OutputDevice(){}

void OutputDevice::setDisplayOn(void) {
	ifScreenDisplay = true;
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

void OutputDevice::printMessage(std::string message1, double whatToPrint1,std::string message2, double whatToPrint2 ) const{
	if(ifScreenDisplay){
		std::cout<<message1<<" ";
		std::cout<<whatToPrint1<<" ";
		std::cout<<message2<<" ";
		std::cout<<whatToPrint2<<"\n";
	}
}


void OutputDevice::printMessage(std::string message, vec whatToPrint) const{
	if(ifScreenDisplay){
		whatToPrint.print(message);
	}
}

void OutputDevice::printMessage(std::string message, rowvec whatToPrint) const{
	if(ifScreenDisplay){
		whatToPrint.print(message);
	}
}

void OutputDevice::printMessage(std::string message, mat whatToPrint) const{
	if(ifScreenDisplay){
		whatToPrint.print(message);
	}
}

void OutputDevice::printList(std::vector<int> list, std::string msg) const{

	assert(!list.empty());

	if(ifScreenDisplay){
		printMessage(msg);

		for (std::vector<int>::const_iterator i = list.begin(); i != list.end(); ++i)
			std::cout << *i << ' ';

		std::cout <<'\n';

	}
}


void OutputDevice::printDesign(const Design &d) const{
	if(ifScreenDisplay){
		d.print();
	}
}

void OutputDevice::printDesign(const DesignForBayesianOptimization &d) const{
	if(ifScreenDisplay){
		d.print();
	}
}

void OutputDevice::printIteration(unsigned int iteration) const{
	if(ifScreenDisplay){
		std::cout<<"\n";
		std::string whatToPrint;
		whatToPrint = "################################# ";
		whatToPrint += "Iteration = ";
		whatToPrint += std::to_string(iteration);
		whatToPrint += " #################################";
		std::cout<<whatToPrint<<"\n";
	}
}


void OutputDevice::openLogFile(void) {
    std::string dateTime = getCurrentDateTime();
	logFile = "LOG" + dateTime + ".dat";

    std::ofstream outFile(logFile);
    if (!outFile) {
        throw std::ios_base::failure("Failed to open the file: " + logFile);
    }

}



void OutputDevice::printBoxConstraints(const vec &lb, const vec &ub) const{

	unsigned int dim = lb.size();
	assert(dim > 0);
	assert(dim == ub.size());

	if(ifScreenDisplay){
		for(unsigned int i=0; i<dim; i++){
			std::cout<<"Parameter("<<i<<") = "<<lb(i)<<"  "<<ub(i)<<"\n";
		}
	}
}

void OutputDevice::printToLogFile(const std::string& content) const {
    assert(!content.empty());
    assert(!logFile.empty());

    if (ifWriteToLogFile) {
        std::ofstream outFile(logFile, std::ios::app); // Open in append mode
        if (!outFile) {
            throw std::ios_base::failure("Failed to open the file: " + logFile);
        }

        outFile << content << "\n";

        if (!outFile) {
            throw std::ios_base::failure("Failed to write to the file: " + logFile);
        }

        outFile.close();
        if (!outFile) {
            throw std::ios_base::failure("Failed to close the file: " + logFile);
        }
    }
}


