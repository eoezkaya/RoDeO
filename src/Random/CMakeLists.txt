set(src_files random_functions.cpp)
add_library(random ${src_files})


target_compile_options(random PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS random COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for random are: ${MAIN_CFLAGS}")

add_custom_command(TARGET random POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)