set(src_files eggholder.cpp himmelblau.cpp standard_test_functions.cpp test_functions.cpp wingweight.cpp griewank2D.cpp)
add_library(testFunctions ${src_files})
target_compile_options(testFunctions PUBLIC -Wall -Wextra -Wpedantic -O2)
target_link_libraries(testFunctions PUBLIC armadillo)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS testFunctions COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for testFunctions are: ${MAIN_CFLAGS}")

add_custom_command(TARGET testFunctions POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})




if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)