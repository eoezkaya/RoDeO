set(src_files correlation_functions.cpp exponential_correlation_function.cpp gaussian_correlation_function.cpp)
add_library(correlationFunctions ${src_files})
target_compile_options(correlationFunctions PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

get_target_property(MAIN_CFLAGS correlationFunctions COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for correlationFunctions are: ${MAIN_CFLAGS}")

add_custom_command(TARGET correlationFunctions POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

