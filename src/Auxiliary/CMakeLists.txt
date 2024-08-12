file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.hpp) 


add_library(auxiliary SHARED ${SOURCES})

target_include_directories(auxiliary PUBLIC INCLUDE)
target_include_directories(auxiliary PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(auxiliary PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
target_link_libraries(auxiliary armadillo::armadillo)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for auxiliary are: ${MAIN_CFLAGS}")

add_custom_command(TARGET auxiliary POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS auxiliary 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
