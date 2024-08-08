cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(cppAPI VERSION 1.0)

# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/INCLUDE)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Set the source files for the shared library
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/API/*.cpp)

# Set the header files for installation
#file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/INCLUDE/*.h)
#find_package(LAPACK REQUIRED)
# Create the shared library
add_library(cppAPI SHARED ${SOURCES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optionally, set the version number for the shared library
set_target_properties(cppAPI PROPERTIES VERSION ${PROJECT_VERSION})

target_link_libraries(cppAPI PRIVATE optimizers)
target_link_libraries(cppAPI PRIVATE objectiveFunctions)
target_link_libraries(cppAPI PRIVATE surrogateModels)
target_link_libraries(cppAPI PRIVATE output)
target_link_libraries(cppAPI PRIVATE externalFunctions)

include(GNUInstallDirs) 


# Install the library and headers
install(TARGETS cppAPI
    #EXPORT cppAPI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Fix the install DIRECTORY command by specifying the correct DESTINATION
install(DIRECTORY ${CMAKE_SOURCE_DIR}/API/INCLUDE/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the targets for use in other projects
#install(EXPORT cppAPI
#    FILE cppAPITargets.cmake
#    NAMESPACE cppAPI::
#    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppAPI
#)

# Include the package configuration helpers
include(CMakePackageConfigHelpers)


# Install the configuration files
#install(FILES
#    "${CMAKE_CURRENT_BINARY_DIR}/cppAPIConfig.cmake"
#    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppAPI
#)
