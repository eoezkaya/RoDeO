set(src_files output.cpp)
add_library(output ${src_files})
target_compile_options(output PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
target_link_libraries(output armadillo::armadillo)
if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS output COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for output are: ${MAIN_CFLAGS}")

add_custom_command(TARGET output POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
