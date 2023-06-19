if(UNIT_TESTS)
	set(src_files kriging_test.cpp multi_level_model_test.cpp surrogate_model_base_test.cpp surrogate_model_tester_test.cpp ggek_test.cpp linear_regression_test.cpp surrogate_model_data_test.cpp)
	add_executable(runTestsSurrogateModels runTests.cpp ${src_files})
	find_package(Armadillo REQUIRED)
	include_directories(${ARMADILLO_INCLUDE_DIRS})
	target_link_libraries(runTestsSurrogateModels gtest_main)
	target_link_libraries(runTestsSurrogateModels ${ARMADILLO_LIBRARIES})
	target_link_libraries(runTestsSurrogateModels   testFunctions)
	target_link_libraries(runTestsSurrogateModels   optimizers)
	target_link_libraries(runTestsSurrogateModels   surrogateModels)
	target_link_libraries(runTestsSurrogateModels   output)
	target_link_libraries(runTestsSurrogateModels   objectiveFunctions)
	target_link_libraries(runTestsSurrogateModels   lhs)
	

	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)	
		target_link_libraries(runTestsSurrogateModels OpenMP::OpenMP_CXX)
	endif()
	
	
	
	

endif(UNIT_TESTS)
