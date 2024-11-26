#ifndef DRIVER_LOGGER
#define DRIVER_LOGGER


#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <ctime>
#include <thread>
#include <iomanip>

namespace Rodop{

class DriverLogger {
private:
	std::ofstream logFile;
	LogLevel currentLogLevel;
	std::mutex logMutex;

	DriverLogger() : currentLogLevel(INFO) {
		logFile.open("driver.log", std::ios::out | std::ios::app);
		if (!logFile.is_open()) {
			std::cerr << "Failed to open log file." << std::endl;
		}
	}

public:
	// Singleton instance
	static DriverLogger& getInstance() {
		static DriverLogger instance;
		return instance;
	}

	// Set log level
	void setLogLevel(LogLevel level) {
		currentLogLevel = level;
	}

	// Log a message
	void log(LogLevel level, const std::string& message) {
		if (level >= currentLogLevel) {
			std::lock_guard<std::mutex> lock(logMutex);
			logFile << getCurrentTime() << " [" << getLevelString(level) << "] "
					<< std::this_thread::get_id() << ": " << message << std::endl;
		}
	}

	// Destructor
	~DriverLogger() {
		if (logFile.is_open()) {
			logFile.close();
		}
	}

private:
	std::string getCurrentTime() {
		std::time_t now = std::time(0);
		std::tm* localtm = std::localtime(&now);
		std::stringstream ss;
		ss << std::put_time(localtm, "%Y-%m-%d %H:%M:%S");
		return ss.str();
	}

	std::string getLevelString(LogLevel level) {
		switch (level) {
		case DEBUG: return "DEBUG";
		case INFO: return "INFO";
		case WARNING: return "WARNING";
		case ERROR: return "ERROR";
		case CRITICAL: return "CRITICAL";
		default: return "UNKNOWN";
		}
	}
};

} /* Namespace Rodop */

#endif
