cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(cppAPI VERSION 1.0)

file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.h ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.h)

add_library(cppAPI SHARED ${SOURCES})

target_include_directories(cppAPI PUBLIC INCLUDE)
target_include_directories(cppAPI PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(cppAPI PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

get_target_property(MAIN_CFLAGS cppAPI COMPILE_OPTIONS)

target_link_libraries(cppAPI random)
target_link_libraries(cppAPI auxiliary)
target_link_libraries(cppAPI design)
target_link_libraries(cppAPI linearAlgebra)
target_link_libraries(cppAPI output)
target_link_libraries(cppAPI optimizers)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET cppAPI POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS cppAPI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
