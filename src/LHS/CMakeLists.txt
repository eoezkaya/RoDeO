set(src_files lhs.cpp)
add_library(lhs ${src_files})

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

target_compile_options(lhs PUBLIC -Wall -Wextra -Wpedantic -O2)

get_target_property(MAIN_CFLAGS lhs COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for lhs are: ${MAIN_CFLAGS}")

add_custom_command(TARGET lhs POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)