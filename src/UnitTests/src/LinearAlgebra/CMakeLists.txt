set(src_files linear_solver_test.cpp matrix_operations_test.cpp vector_operations_test.cpp)
add_library(linearAlgebraTest ${src_files})

add_executable(runTests runTests.cpp linear_solver_test.cpp matrix_operations_test.cpp vector_operations_test.cpp)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(runTests gtest_main)
target_link_libraries(runTests ${ARMADILLO_LIBRARIES})
target_link_libraries(runTests   linearAlgebra)
target_link_libraries(runTests   auxiliary)
