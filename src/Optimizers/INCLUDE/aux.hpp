#ifndef OPTIMIZATION_AUX
#define OPTIMIZATION_AUX
#include <string>


namespace Rodop{

int getRandomInteger(int a, int b);
double getRandomDouble(double a, double b);
std::string generateProcessStartMessage(int borderLength = 100);
std::string generateProcessEndMessage(int borderLength = 100);
std::string getCurrentDateTime();

std::string generateFormattedString(std::string msg, char c, int totalLength);
std::string generateFormattedString(std::string& content);

}


#endif
