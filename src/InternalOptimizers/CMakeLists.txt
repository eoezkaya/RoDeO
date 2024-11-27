# Set the source files
set(src_files 
    ea_optimizer.cpp 
    ea_individual.cpp 
    ea_population.cpp 
    general_purpose_optimizer.cpp 
)

# Create a library target named "internalOptimizers"
add_library(internalOptimizers ${src_files})

# Set compiler options for this target
target_compile_options(internalOptimizers PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

# Retrieve the compile options for debugging purposes
get_target_property(MAIN_CFLAGS internalOptimizers COMPILE_OPTIONS)

# Handle assertions based on the ASSERTS option
if(ASSERTS)
    # Add definitions related to asserts if needed
else()
    # Disable assertions in release mode
    add_definitions(-DNDEBUG)
endif(ASSERTS)

# Link the linear algebra library to the internalOptimizers target
target_link_libraries(internalOptimizers linearAlgebra)

# Find and link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(internalOptimizers OpenMP::OpenMP_CXX)
    target_compile_definitions(internalOptimizers PUBLIC OPENMP_SUPPORT)
endif()

# Output the compiler flags used for building internalOptimizers
message("-- Target compiler flags for internalOptimizers are: ${MAIN_CFLAGS}")

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for surrogateModels are: ${MAIN_CFLAGS}")

add_custom_command(TARGET internalOptimizers POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

