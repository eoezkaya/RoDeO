cmake_minimum_required(VERSION 3.24.2)
project(rodeo VERSION 1.3)

# Add all required module directories
add_subdirectory(Design)
add_subdirectory(Bounds)
add_subdirectory(CorrelationFunctions)
add_subdirectory(LinearAlgebra)
add_subdirectory(ObjectiveFunctions)
add_subdirectory(Optimizers)
add_subdirectory(InternalOptimizers)
add_subdirectory(SurrogateModels)
add_subdirectory(API)  # The API directory contains the shared library


