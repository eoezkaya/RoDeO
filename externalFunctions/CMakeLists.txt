# CMakeLists.txt

# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.12)

# Project name
project(externalFunctions)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCE_FILES *.cpp)


# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/INCLUDE)

# Create the shared library
add_library(externalFunctions STATIC ${SOURCE_FILES})

# Set the output directory for the library
set_target_properties(externalFunctions PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
install(TARGETS "externalFunctions" EXPORT "rodeoTargets")
