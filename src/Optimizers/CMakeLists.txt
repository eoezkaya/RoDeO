set(src_files design.cpp ea_optimizer.cpp general_purpose_optimizer.cpp gradient_optimizer.cpp optimization.cpp)
add_library(optimizers ${src_files})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)	
	add_definitions(-DOPENMP_SUPPORT)
	target_compile_options(optimizers PUBLIC -Wall -Wextra -Wpedantic -O2 -fopenmp)
	
else()
	target_compile_options(optimizers PUBLIC -Wall -Wextra -Wpedantic -O2)
		
endif()

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS optimizers COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for optimizers are: ${MAIN_CFLAGS}")

add_custom_command(TARGET optimizers POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})
