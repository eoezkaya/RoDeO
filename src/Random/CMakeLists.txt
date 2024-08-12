file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp)


add_library(random SHARED ${SOURCES})

target_include_directories(random PUBLIC INCLUDE)
target_include_directories(random PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)
target_link_libraries(random PUBLIC armadillo::armadillo auxiliary)
target_compile_options(random PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

get_target_property(MAIN_CFLAGS random COMPILE_OPTIONS)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for random are: ${MAIN_CFLAGS}")

add_custom_command(TARGET random POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})


if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS random
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

