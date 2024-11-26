file(GLOB src_files *.cpp)
add_library(optimizers ${src_files})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(optimizers OpenMP::OpenMP_CXX)
    target_compile_definitions(optimizers PUBLIC OPENMP_SUPPORT)
endif()


target_compile_options(optimizers PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
		

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)


target_link_libraries(optimizers linearAlgebra)
target_link_libraries(optimizers objectiveFunctions)
target_link_libraries(optimizers design)
target_link_libraries(optimizers surrogateModels)




get_target_property(MAIN_CFLAGS optimizers COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for optimizers are: ${MAIN_CFLAGS}")

add_custom_command(TARGET optimizers POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
	add_subdirectory(./UnitTests)
endif(UNIT_TESTS)
