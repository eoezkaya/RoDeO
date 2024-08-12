file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.hpp) 


add_library(correlationFunctions SHARED ${SOURCES})

target_include_directories(correlationFunctions PUBLIC INCLUDE)
target_include_directories(correlationFunctions PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(correlationFunctions PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)
target_link_libraries(correlationFunctions armadillo::armadillo auxiliary linearAlgebra)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for auxiliary are: ${MAIN_CFLAGS}")

add_custom_command(TARGET correlationFunctions POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS correlationFunctions
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
