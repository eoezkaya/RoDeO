set(src_files linear_solver.cpp matrix_operations.cpp vector_operations.cpp)
add_library(linearAlgebra ${src_files})

target_compile_options(linearAlgebra PUBLIC -Wall -Wextra -Wpedantic -O2)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS linearAlgebra COMPILE_OPTIONS)

target_link_libraries(linearAlgebra random)
target_link_libraries(linearAlgebra bounds)
target_link_libraries(linearAlgebra auxiliary)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET linearAlgebra POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

add_subdirectory(./UnitTests)
