file(GLOB src_files *.cpp)
add_library(optimizers ${src_files})

target_compile_options(optimizers PRIVATE ${OpenMP_CXX_FLAGS})
if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

target_link_libraries(optimizers linearAlgebra)
target_link_libraries(optimizers   auxiliary)
target_link_libraries(optimizers   output)

get_target_property(MAIN_CFLAGS optimizers COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for optimizers are: ${MAIN_CFLAGS}")

add_custom_command(TARGET optimizers POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
