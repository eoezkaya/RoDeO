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


target_link_libraries(cppAPI optimizers)
target_link_libraries(cppAPI objectiveFunctions)
target_link_libraries(cppAPI surrogateModels)
target_link_libraries(cppAPI output)

install(TARGETS cppAPI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/API/INCLUDE/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
