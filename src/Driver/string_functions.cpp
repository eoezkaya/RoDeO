#include "./INCLUDE/string_functions.hpp"
#include <string>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>
using std::string;


bool isEqual(const string& str1, const string& str2) {

	assert(!str1.empty());
	assert(!str2.empty());

    // Check if the strings are of the same length
    if (str1.length() != str2.length()) {
        return false;
    }

    // Compare characters in a case-insensitive manner
    for (size_t i = 0; i < str1.length(); ++i) {
        if (tolower(str1[i]) != tolower(str2[i])) {
            return false;
        }
    }

    return true;
}



bool isEmpty(std::string inputStr){

	if(inputStr.empty()){

		return true;
	}
	else{

		return false;
	}

}


bool isNotEmpty(std::string inputStr){

	if(inputStr.empty()){

		return false;
	}
	else{

		return true;
	}

}

bool checkIfOn(const std::string& message) {
    // Use unordered_set for faster lookups
    static const std::unordered_set<std::string> positiveMessages = {"yes", "y", "on", "true", "enable", "enabled"};

    // Convert input message to lower case for case-insensitive comparison
    std::string lowerMessage = message;
    std::transform(lowerMessage.begin(), lowerMessage.end(), lowerMessage.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // Check if the transformed message is in the set of positive messages
    return positiveMessages.find(lowerMessage) != positiveMessages.end();
}

bool checkIfOff(const std::string& message) {
    // Use unordered_set for faster lookups of "off" messages
    static const std::unordered_set<std::string> offMessages = {"no", "n", "off", "false", "disable", "disabled"};

    // Convert input message to lower case for case-insensitive comparison
    std::string lowerMessage = message;
    std::transform(lowerMessage.begin(), lowerMessage.end(), lowerMessage.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // Check if the transformed message is in the set of off messages
    return offMessages.find(lowerMessage) != offMessages.end();
}

std::string removeSpacesFromString(std::string inputString){

	inputString.erase(std::remove_if(inputString.begin(), inputString.end(), ::isspace), inputString.end());
	return inputString;
}





std::string extractContentBetweenKeys(const std::string& input, const std::string& key) {
    std::string startTag = "<" + key + ">";
    std::string endTag = "</" + key + ">";
    std::string result;

    // Find the position of the start tag
    size_t startPos = input.find(startTag);
    if (startPos == std::string::npos) {
        throw std::runtime_error("Start tag not found for the key: " + key);
    }

    // Find the position of the end tag
    size_t endPos = input.find(endTag, startPos + startTag.length());
    if (endPos == std::string::npos) {
        throw std::runtime_error("End tag not found for the key: " + key);
    }

    // Check for any additional occurrences of the start tag before the end tag
    size_t nextStartPos = input.find(startTag, startPos + startTag.length());
    if (nextStartPos != std::string::npos) {
        throw std::runtime_error("Multiple occurrences of the key found: " + key);
    }

    // Extract content between start and end tag
    size_t contentStart = startPos + startTag.length();
    size_t contentLength = endPos - contentStart;
    result = input.substr(contentStart, contentLength);

    return result;
}

std::vector<std::string> extractContents(const std::string& input, const std::string& key) {
    std::vector<std::string> result;
    std::string startTag = "<" + key + ">";
    std::string endTag = "</" + key + ">";

    size_t startPos = input.find(startTag);
    while (startPos != std::string::npos) {
        size_t endPos = input.find(endTag, startPos + startTag.length());
        if (endPos != std::string::npos) {
            // Extract the content between start and end tags
            std::string content = input.substr(startPos + startTag.length(), endPos - (startPos + startTag.length()));
            result.push_back(content);

            // Search for the next occurrence of startTag
            startPos = input.find(startTag, endPos + endTag.length());
        } else {
            break; // No matching endTag found, exit loop
        }
    }

    return result;
}

std::string toLower(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    return lower_str;
}

