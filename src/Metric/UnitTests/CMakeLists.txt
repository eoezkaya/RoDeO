if(UNIT_TESTS)
	set(src_files metric_test.cpp)
	add_executable(runTestsMetric runTests.cpp ${src_files})
	find_package(Armadillo REQUIRED)
	include_directories(${ARMADILLO_INCLUDE_DIRS})
	target_link_libraries(runTestsMetric gtest_main)
	target_link_libraries(runTestsMetric ${ARMADILLO_LIBRARIES})
	target_link_libraries(runTestsMetric   metric)
	target_link_libraries(runTestsMetric   optimizers)
	target_link_libraries(runTestsMetric   output)
	target_link_libraries(runTestsMetric   auxiliary)
	
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)	
		target_link_libraries(runTestsMetric OpenMP::OpenMP_CXX)
	endif()
	
	
endif(UNIT_TESTS)
