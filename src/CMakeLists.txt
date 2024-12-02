cmake_minimum_required(VERSION 3.24.2)

execute_process(
    COMMAND ${CMAKE_COMMAND} .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../externalFunctions
)

execute_process(
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../externalFunctions
)

# Add externalFunctions library
add_subdirectory(../externalFunctions externalFunctions)

# Link with externalFunctions library
target_link_libraries(${PROJECT_NAME} externalFunctions)


project(rodeo VERSION 1.3)
add_executable(${PROJECT_NAME}  Rodeo.cpp)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra -Wpedantic -O2)


add_subdirectory(Design)
add_subdirectory(Bounds)
add_subdirectory(CorrelationFunctions)
add_subdirectory(Driver)
add_subdirectory(LinearAlgebra)
add_subdirectory(ObjectiveFunctions)
add_subdirectory(Optimizers)
add_subdirectory(InternalOptimizers)
add_subdirectory(SurrogateModels)
add_subdirectory(API)


target_link_libraries(${PROJECT_NAME}  bounds)
target_link_libraries(${PROJECT_NAME}  correlationFunctions)
target_link_libraries(${PROJECT_NAME}  driver)
target_link_libraries(${PROJECT_NAME}  linearAlgebra)
target_link_libraries(${PROJECT_NAME}  objectiveFunctions)
target_link_libraries(${PROJECT_NAME}  optimizers)
target_link_libraries(${PROJECT_NAME}  internalOptimizers)
target_link_libraries(${PROJECT_NAME}  surrogateModels)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)	
	target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
	add_definitions(-DOPENMP_SUPPORT)
endif()


get_target_property(MAIN_CFLAGS ${PROJECT_NAME} COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags are: ${MAIN_CFLAGS}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})