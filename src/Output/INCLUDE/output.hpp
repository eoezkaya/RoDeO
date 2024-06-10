#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include<string>
#include "../../LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "../../Optimizers/INCLUDE/design.hpp"

using std::string;

void printIntro(void);
std::string getCurrentDateTime();
std::string generateProcessStartMessage();
std::string generateProcessEndMessage();
std::string generateFormattedString(std::string& content);
std::string generateFormattedString(std::string msg, char c, int numEquals);

class OutputDevice{


public:

	bool ifScreenDisplay = false;
	bool ifWriteToLogFile = true;
	std::string logFile = "LOG.dat";

	OutputDevice();
	void setDisplayOn(void);

	void printMessage(string) const;
	void printErrorMessageAndAbort(string message) const;

	void openLogFile(void);

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

	void printToLogFile(const std::string& content) const;
	void printContentToLogFile(const std::string& msg, double value) const;









};

#endif

