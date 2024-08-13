file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.hpp)

add_library(design SHARED ${SOURCES})
target_include_directories(design PUBLIC INCLUDE)
target_include_directories(design PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(design PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

target_link_libraries(design random)
target_link_libraries(design auxiliary)
target_link_libraries(design linearAlgebra)
target_link_libraries(design armadillo::armadillo)

message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET design POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS design 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

