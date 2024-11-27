set(src_files 
	kriging_training.cpp 
	linear_regression.cpp 
	surrogate_model.cpp 
	surrogate_model_data.cpp 
	surrogate_model_tester.cpp
)

# Create a library target named "surrogateModels"
add_library(surrogateModels ${src_files})


target_compile_options(surrogateModels PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
		

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)


target_link_libraries(surrogateModels correlationFunctions)
target_link_libraries(surrogateModels internalOptimizers)
target_link_libraries(surrogateModels linearAlgebra)

# Find and link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(surrogateModels OpenMP::OpenMP_CXX)
    target_compile_definitions(surrogateModels PUBLIC OPENMP_SUPPORT)
endif()


get_target_property(MAIN_CFLAGS surrogateModels COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for surrogateModels are: ${MAIN_CFLAGS}")

add_custom_command(TARGET surrogateModels POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)