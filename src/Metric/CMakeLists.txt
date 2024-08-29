set(src_files metric.cpp)
add_library(metric ${src_files})
target_compile_options(metric PRIVATE ${OpenMP_CXX_FLAGS})

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
