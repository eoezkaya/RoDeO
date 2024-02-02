add_library(auxiliary auxiliary_functions.cpp string_functions.cpp print.cpp)
target_compile_options(auxiliary PUBLIC -Wall -Wextra -Wpedantic -O2)

get_target_property(MAIN_CFLAGS auxiliary COMPILE_OPTIONS)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for auxiliary are: ${MAIN_CFLAGS}")

add_custom_command(TARGET auxiliary POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
