set(src_files objective_function.cpp constraint_functions.cpp)
add_library(objectiveFunctions ${src_files})

target_compile_options(objectiveFunctions PUBLIC -Wall -Wextra -Wpedantic -O2)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS objectiveFunctions COMPILE_OPTIONS)

add_definitions(-DNDEBUG)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for objectiveFunctions are: ${MAIN_CFLAGS}")

add_custom_command(TARGET objectiveFunctions POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})
