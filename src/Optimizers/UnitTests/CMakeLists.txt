if(UNIT_TESTS)
	set(src_files ea_optimizer_test.cpp general_purpose_optimizer_test.cpp gradient_optimizer_test.cpp optimization_test.cpp design_test.cpp)
	add_executable(runTestsOptimizers runTests.cpp ${src_files})
	find_package(Armadillo REQUIRED)
	include_directories(${ARMADILLO_INCLUDE_DIRS})
	target_link_libraries(runTestsOptimizers gtest_main)
	target_link_libraries(runTestsOptimizers ${ARMADILLO_LIBRARIES})
	target_link_libraries(runTestsOptimizers   testFunctions)
	target_link_libraries(runTestsOptimizers   optimizers)
	target_link_libraries(runTestsOptimizers   surrogateModels)
	target_link_libraries(runTestsOptimizers   output)
	target_link_libraries(runTestsOptimizers   objectiveFunctions)
	
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)	
		target_link_libraries(runTestsOptimizers OpenMP::OpenMP_CXX)
	endif()
	
	
	
	

endif(UNIT_TESTS)
