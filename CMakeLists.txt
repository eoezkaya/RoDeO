cmake_minimum_required(VERSION 3.24.2)
project(rodeo VERSION 1.0)

# Enable CPack
include(CPack)

# Set CPack variables
set(CPACK_PACKAGE_NAME "rodeo")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")

option(UNIT_TESTS "UNITS_TESTS" OFF)
option(ASSERTS "ASSERTS" OFF)

# Add externalFunctions library
add_subdirectory(externalFunctions externalFunctions)

if(UNIT_TESTS)
    set(ASSERTS ON)
    message("-- Building with Unit Tests")
    add_definitions(-DUNIT_TESTS)

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
else()
    message("-- Building without Unit Tests")
endif()

if(ASSERTS)
    message("-- Asserts are on")
else()
    message("-- Asserts are off")
    add_definitions(-DNDEBUG)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define MAIN_CFLAGS with necessary flags
set(MAIN_CFLAGS -Wall -Wextra -Wpedantic -O2)
add_compile_options(${MAIN_CFLAGS})

# Include the directories
include_directories(${CMAKE_SOURCE_DIR}/src/INCLUDE)

# Add subdirectories located under src/
add_subdirectory(src/Auxiliary)
add_subdirectory(src/Bounds)
add_subdirectory(src/CorrelationFunctions)
add_subdirectory(src/Driver)
add_subdirectory(src/LinearAlgebra)
add_subdirectory(src/Metric)
add_subdirectory(src/ObjectiveFunctions)
add_subdirectory(src/Optimizers)
add_subdirectory(src/Output)
add_subdirectory(src/Random)
add_subdirectory(src/SurrogateModels)
add_subdirectory(src/TestFunctions)
add_subdirectory(src/API)

# Define the main executable target
add_executable(rodeo src/Rodeo.cpp)

# Link libraries to the main executable target
target_link_libraries(rodeo PRIVATE
    auxiliary bounds correlationFunctions driver
    linearAlgebra metric objectiveFunctions optimizers
    output random surrogateModels testFunctions
    externalFunctions
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(rodeo PRIVATE ${ARMADILLO_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rodeo PRIVATE OpenMP::OpenMP_CXX)
    add_definitions(-DOPENMP_SUPPORT)
endif()

get_target_property(MAIN_CFLAGS rodeo COMPILE_OPTIONS)
string(REPLACE ";" " " MAIN_CFLAGS "${MAIN_CFLAGS}")
message("-- Target compiler flags are: ${MAIN_CFLAGS}")

add_custom_command(TARGET rodeo POST_BUILD
    COMMAND echo "Built with the flags: ${MAIN_CFLAGS}")

# Install the executable
install(TARGETS rodeo DESTINATION bin)
