set(src_files ggek.cpp kriging_training.cpp linear_regression.cpp multi_level_model.cpp surrogate_model.cpp surrogate_model_data.cpp surrogate_model_tester.cpp)
add_library(surrogateModels ${src_files})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)	
	add_definitions(-DOPENMP_SUPPORT)
	target_compile_options(surrogateModels PUBLIC -Wall -Wextra -Wpedantic -O2 -fopenmp)
	
else()
	target_compile_options(surrogateModels PUBLIC -Wall -Wextra -Wpedantic -O2)
		
endif()


if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)




target_link_libraries(surrogateModels correlationFunctions)
target_link_libraries(surrogateModels auxiliary)
target_link_libraries(surrogateModels linearAlgebra)
target_link_libraries(surrogateModels optimizers)
target_link_libraries(surrogateModels metric)

get_target_property(MAIN_CFLAGS surrogateModels COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for surrogateModels are: ${MAIN_CFLAGS}")

add_custom_command(TARGET surrogateModels POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)