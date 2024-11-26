#include "./INCLUDE/aux.hpp"
#include <random>
#include <sstream>
#include <chrono>
#include <iomanip>
namespace Rodop{


int getRandomInteger(int a, int b) {
	// Static generator to avoid reseeding on each function call
	static std::random_device rd;
	static std::mt19937 gen(rd());

	// Define the distribution between a and b (inclusive)
	std::uniform_int_distribution<> dis(a, b);

	// Generate and return the random integer
	return dis(gen);
}

double getRandomDouble(double a, double b) {
	// Create a random device and seed the generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define the distribution between a and b (inclusive)
	std::uniform_real_distribution<> dis(a, b);

	// Generate and return the random double
	return dis(gen);
}

std::string generateProcessStartMessage(int borderLength) {
	std::string dateTime = getCurrentDateTime();
	std::ostringstream oss;

	// Create a border with the specified length
	std::string border(borderLength, '=');

	// Construct the message
	oss << border << "\n"
			<< "Process start: \"" << dateTime << "\"\n"
			<< border << "\n";

	return oss.str();
}


std::string generateProcessEndMessage(int borderLength) {
		std::string dateTime = getCurrentDateTime();
		std::ostringstream oss;

		std::string border(borderLength, '=');
		oss << border <<"\n"
				<< "Process end: \"" << dateTime << "\"\n"
				<< border <<"\n";

		return oss.str();
	}

std::string getCurrentDateTime(){
	auto now = std::chrono::system_clock::now();
	std::time_t now_time = std::chrono::system_clock::to_time_t(now);
	std::tm* now_tm = std::localtime(&now_time);

	std::ostringstream oss;
	oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
	return oss.str();
}

std::string generateFormattedString(std::string msg, char c, int totalLength) {
	if (totalLength < 0) {
		throw std::invalid_argument("Number of characters must be non-negative.");
	}

	if(msg.length()%2 == 1){
		msg+=" ";
	}

	int numEquals = static_cast<int>( (totalLength - msg.length() - 2)/2);


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

	int padding = static_cast<int>((totalWidth - content.length()) / 2);
	int extraPadding = (totalWidth - content.length()) % 2; // handle odd lengths for perfect centering

	oss << std::string(padding, ' ') << content << std::string(padding + extraPadding, ' ') << "\n";
	oss << border;

	return oss.str();
}






}
