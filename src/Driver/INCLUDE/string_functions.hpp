#ifndef STRING_FUNCTIONS_HPP
#define STRING_FUNCTIONS_HPP

#include <vector>
#include <sstream>
#include <string>
#include <cassert>
#include <algorithm>



bool isEqual(const std::string& str1, const std::string& str2);
bool checkIfOn(const std::string& message);
bool checkIfOff(const std::string& message);

std::string removeSpacesFromString(std::string );


std::string extractContentBetweenKeys(const std::string& input, const std::string& key);
std::vector<std::string> extractContents(const std::string& input, const std::string& key);


template <typename T>
std::string convertToString(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

void readFileToaString(std::string filename, std::string & stringCompleteFile);

std::string toLower(const std::string& str);

#endif
