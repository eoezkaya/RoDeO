cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(cppAPI VERSION 1.0)


# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/INCLUDE)

# Set the source files for the shared library
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/API/*.cpp)

# Create the shared library
add_library(cppAPI SHARED ${SOURCES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optionally, set the version number for the shared library
set_target_properties(cppAPI PROPERTIES VERSION ${PROJECT_VERSION})


target_link_libraries(cppAPI PRIVATE optimizers)
target_link_libraries(cppAPI PRIVATE objectiveFunctions)
target_link_libraries(cppAPI PRIVATE surrogateModels)
target_link_libraries(cppAPI PRIVATE output)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/INCLUDE/ DESTINATION include)
install( TARGETS "cppAPI" EXPORT "cppAPITargets")
install(EXPORT "cppAPITargets" DESTINATION "cmake")
