add_library(bounds bounds.cpp)

target_compile_options(bounds PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

get_target_property(MAIN_CFLAGS bounds COMPILE_OPTIONS)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for bounds are: ${MAIN_CFLAGS}")

add_custom_command(TARGET bounds POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

target_link_libraries(bounds random armadillo::armadillo)


if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
