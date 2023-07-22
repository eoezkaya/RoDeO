#!/usr/bin/env python

import os

print('Running unit tests for the module: Auxiliary')
print('#############################################################')
os.chdir("../build/Auxiliary/UnitTests")
os.system("./runTestsAuxiliary")

print('Running unit tests for the module: Bounds')
print('#############################################################')
os.chdir("../../../build/Bounds/UnitTests")
os.system("./runBoundsTest")

print('Running unit tests for the module: CorrelationFunctions')
print('#############################################################')
os.chdir("../../../build/CorrelationFunctions/UnitTests")
os.system("./runTestsCorrelationFunctions")

print('Running unit tests for the module: Metric')
print('#############################################################')
os.chdir("../../../build/Metric/UnitTests")
os.system("./runTestsMetric")

print('Running unit tests for the module: LinearAlgebra')
print('#############################################################')
os.chdir("../../../build/LinearAlgebra/UnitTests")
os.system("./runTestsLinearAlgebra")


print('Running unit tests for the module: ObjectiveFunctions')
print('#############################################################')
os.chdir("../../../build/ObjectiveFunctions/UnitTests")
os.system("./runTestsObjectiveFunctions")

print('Running unit tests for the module: TestFunctions')
print('#############################################################')
os.chdir("../../../build/TestFunctions/UnitTests")
os.system("./runTestsTestFunctions")

print('Running unit tests for the module: Optimizers')
print('#############################################################')
os.chdir("../../../build/Optimizers/UnitTests")
os.system("./runTestsOptimizers")

print('Running unit tests for the module: SurrogateModels')
print('#############################################################')
os.chdir("../../../build/SurrogateModels/UnitTests")
os.system("./runTestsSurrogateModels")

print('Running unit tests for the module: Driver')
print('#############################################################')
os.chdir("../../../build/Driver/UnitTests")
os.system("./runTestsDriver")


