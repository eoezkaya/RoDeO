if(UNIT_TESTS)
	set(src_files configkey_test.cpp driver_test.cpp)
	add_executable(runTestsDriver runTests.cpp ${src_files})
	find_package(Armadillo REQUIRED)
	include_directories(${ARMADILLO_INCLUDE_DIRS})
	target_link_libraries(runTestsDriver gtest_main)
	target_link_libraries(runTestsDriver ${ARMADILLO_LIBRARIES})
	target_link_libraries(runTestsDriver   driver)
	target_link_libraries(runTestsDriver   testFunctions)
	target_link_libraries(runTestsDriver   optimizers)
	target_link_libraries(runTestsDriver   surrogateModels)
	target_link_libraries(runTestsDriver   output)
	target_link_libraries(runTestsDriver   objectiveFunctions)
	
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)	
		target_link_libraries(runTestsDriver OpenMP::OpenMP_CXX)
	endif()
	
	
	
	

endif(UNIT_TESTS)
