cmake_minimum_required(VERSION 3.24.2)

project(rodeo LANGUAGES CXX VERSION 1.0)

find_package(Armadillo REQUIRED 12.6.4)
include_directories(${ARMADILLO_INCLUDE_DIRS})

option(UNIT_TESTS "UNITS_TESTS" OFF)
option(ASSERTS "ASSERTS" OFF)
add_executable(${PROJECT_NAME}  Rodeo.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)	
	add_definitions(-DOPENMP_SUPPORT)
	link_libraries(OpenMP::OpenMP_CXX)
endif()

if(WIN32)
	set(CMAKE_CXX_FLAGS "-Wall -O2")
else()
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O2 -fPIC")
endif()



# Add externalFunctions library
add_subdirectory(../externalFunctions externalFunctions)

# Link with externalFunctions library
target_link_libraries(${PROJECT_NAME} externalFunctions)


if(UNIT_TESTS)
	set(ASSERTS ON)	
	message("-- Building with Unit Tests")
	add_definitions(-DUNIT_TESTS)
    
	include(FetchContent)
	FetchContent_Declare(
  	googletest
# Specify the commit you depend on and update it regularly.
 	 URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
	)
# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
	
else()
    message("-- Building without Unit Tests")
endif(UNIT_TESTS)

if(ASSERTS)
    message("-- Asserts are on")
else()
    message("-- Asserts are off")
    add_definitions(-DNDEBUG)
endif(ASSERTS)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

add_subdirectory(Auxiliary)
add_subdirectory(Bounds)
add_subdirectory(CorrelationFunctions)
add_subdirectory(Driver)
add_subdirectory(LinearAlgebra)
add_subdirectory(Metric)
add_subdirectory(ObjectiveFunctions)
add_subdirectory(Optimizers)
add_subdirectory(Output)
add_subdirectory(Random)
add_subdirectory(SurrogateModels)
add_subdirectory(TestFunctions)
add_subdirectory(API)
set(CPACK_INSTALL_CMAKE_PROJECTS "API;cppAPI;ALL;/")
include(CPack)

target_link_libraries(${PROJECT_NAME}  auxiliary)
target_link_libraries(${PROJECT_NAME}  bounds)
target_link_libraries(${PROJECT_NAME}  correlationFunctions)
target_link_libraries(${PROJECT_NAME}  driver)
target_link_libraries(${PROJECT_NAME}  linearAlgebra)
target_link_libraries(${PROJECT_NAME}  metric)
target_link_libraries(${PROJECT_NAME}  objectiveFunctions)
target_link_libraries(${PROJECT_NAME}  optimizers)
target_link_libraries(${PROJECT_NAME}  output)
target_link_libraries(${PROJECT_NAME}  random)
target_link_libraries(${PROJECT_NAME}  surrogateModels)
target_link_libraries(${PROJECT_NAME}  testFunctions)
target_link_libraries(${PROJECT_NAME}  externalFunctions)

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

get_target_property(MAIN_CFLAGS ${PROJECT_NAME} COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags are: ${MAIN_CFLAGS}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})
