file(GLOB src_files *.cpp)
add_library(design ${src_files})

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

target_link_libraries(design linearAlgebra)

target_compile_options(design PUBLIC -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wfloat-equal -O2 -fPIC)


get_target_property(MAIN_CFLAGS design COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for optimizers are: ${MAIN_CFLAGS}")

add_custom_command(TARGET design POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
