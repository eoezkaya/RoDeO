file(GLOB SOURCES *.cpp)
file(GLOB HDRS INCLUDE/*.hpp ${CMAKE_SOURCE_DIR}/src/INCLUDE/*.hpp)

add_library(optimizers SHARED ${SOURCES})
target_include_directories(optimizers PUBLIC INCLUDE)
target_include_directories(optimizers PUBLIC ${CMAKE_SOURCE_DIR}/src/INCLUDE)

target_compile_options(optimizers PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)

if(NOT ASSERTS)
    add_definitions(-DNDEBUG)
endif()

get_target_property(MAIN_CFLAGS optimizers COMPILE_OPTIONS)

target_link_libraries(optimizers random)
target_link_libraries(optimizers auxiliary)
target_link_libraries(optimizers design)
target_link_libraries(optimizers linearAlgebra)
target_link_libraries(optimizers output)
target_link_libraries(optimizers correlationFunctions)
target_link_libraries(optimizers externalFunctions)
target_link_libraries(optimizers armadillo::armadillo)


# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET optimizers POST_BUILD
COMMAND echo built with the flags: ${MAIN_CFLAGS})

if(UNIT_TESTS)
add_subdirectory(./UnitTests)
endif(UNIT_TESTS)

install(TARGETS optimizers 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

