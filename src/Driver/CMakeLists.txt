file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.hpp)

add_library(driver SHARED ${SOURCES})
target_include_directories(driver PUBLIC INCLUDE)
target_include_directories(driver PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(driver PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

get_target_property(MAIN_CFLAGS driver COMPILE_OPTIONS)

target_link_libraries(driver auxiliary)
target_link_libraries(driver linearAlgebra)
target_link_libraries(driver optimizers)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET driver POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS driver
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

