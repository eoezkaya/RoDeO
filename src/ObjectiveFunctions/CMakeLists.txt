set(src_files objective_function.cpp constraint_functions.cpp)
add_library(objectiveFunctions ${src_files})

target_link_libraries(objectiveFunctions PUBLIC armadillo)
target_compile_options(objectiveFunctions PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)

get_target_property(MAIN_CFLAGS objectiveFunctions COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for objectiveFunctions are: ${MAIN_CFLAGS}")

add_custom_command(TARGET objectiveFunctions POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)


find_library(EXTERNAL_FUNCTIONS_LIBRARY externalFunctions
    PATHS ../../externalFunctions/lib
    NO_DEFAULT_PATH
)

# Check if the library is found
if(NOT EXTERNAL_FUNCTIONS_LIBRARY)
    message(FATAL_ERROR "externalFunctions library not found.")
endif()

# Link the executable with the externalFunctions library
target_link_libraries(objectiveFunctions PRIVATE ${EXTERNAL_FUNCTIONS_LIBRARY})



