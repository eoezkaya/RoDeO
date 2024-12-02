# Allow the user to specify a custom OpenBLAS directory
option(USE_CUSTOM_OPENBLAS "Use custom OpenBLAS installation" OFF)
set(OPENBLAS_DIR "" CACHE PATH "Path to the custom OpenBLAS installation")

file(GLOB src_files *.cpp)

add_library(linearAlgebra ${src_files})

target_compile_options(linearAlgebra PUBLIC -Wall -Wextra -Wpedantic -O2 -fPIC)


get_target_property(MAIN_CFLAGS linearAlgebra COMPILE_OPTIONS)
message("-- Target compiler flags for linearAlgebra are: ${MAIN_CFLAGS}")

add_custom_command(TARGET linearAlgebra POST_BUILD
    COMMAND echo built with the flags: ${MAIN_CFLAGS})

# Find and link OpenBLAS library
if(USE_CUSTOM_OPENBLAS AND OPENBLAS_DIR)
    message(STATUS "Using custom OpenBLAS from: ${OPENBLAS_DIR}")
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${OPENBLAS_DIR}/include)
    find_library(OpenBLAS_LIB NAMES openblas PATHS ${OPENBLAS_DIR}/lib)
else()
    message(STATUS "Using system OpenBLAS")
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h)
    find_library(OpenBLAS_LIB NAMES openblas)
endif()


if(OpenBLAS_LIB AND OPENBLAS_INCLUDE_DIR)
    target_link_libraries(linearAlgebra PUBLIC ${OpenBLAS_LIB})
    target_include_directories(linearAlgebra PUBLIC ${OPENBLAS_INCLUDE_DIR})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIB}")
    add_definitions(-DOPENBLAS)
else()
    message(WARNING "OpenBLAS not found. The project will build without OpenBLAS support.")
endif()
