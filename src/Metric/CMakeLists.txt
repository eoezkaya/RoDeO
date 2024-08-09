set(src_files metric.cpp)
add_library(metric ${src_files})
target_link_libraries(metric armadillo::armadillo)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)	
	add_definitions(-DOPENMP_SUPPORT)
	target_compile_options(metric PUBLIC -Wall -Wextra -Wpedantic -O2 -fopenmp -fPIC)
	
else()
	target_compile_options(metric PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
		
endif()

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)


get_target_property(MAIN_CFLAGS metric COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for metric are: ${MAIN_CFLAGS}")

add_custom_command(TARGET metric POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
