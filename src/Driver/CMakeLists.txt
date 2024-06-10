file(GLOB src_files *.cpp)
add_library(driver ${src_files})
target_link_libraries(driver bounds)


if(ASSERTS)
  
else()
    add_definitions(-DNDEBUG)
endif(ASSERTS)


target_compile_options(driver PUBLIC -Wall -Wextra -Wpedantic -O2 )

get_target_property(MAIN_CFLAGS driver COMPILE_OPTIONS)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for driver are: ${MAIN_CFLAGS}")

add_custom_command(TARGET driver POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)