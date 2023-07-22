# RoDeO
RoDeo (Robust Design Optimization Package) is a package for simulation based global design optimization. It is specifically designed
for scientific/engineering applications, in which the objective function and constraints are evaluated by computationally expensive simulations. 

## Features of the RoDeO Package:
- Surrogate model based Efficient Global Optimization (EGO) strategy
- Full data-driven approach
- Easy and efficient treatment of inequality constraints
- Gradient and tangent enhaced surrogate modeling 
 
 
## External Libraries
 - RoDeO uses the linear algebra library Armadillo. Therefore, before compiling RoDeO sources, Armadillo must be installed in the system. To download Armadillo and
 see the documentation please refer to https://arma.sourceforge.net/ 
 - To build and run unit tests (optional), you will require the testing and mocking library GoogleTest installed in your system. For details, please refer to 
 https://github.com/google/googletest
 
## Build Support

- RoDeO uses cmake utility for the compilation and the build process. 
- First go to the src directory and call cmake:
```
cmake -B ../build
```
   
- To build with unit tests use option UNIT_TESTS=ON: 
```
cmake -DUNIT_TESTS=ON -B ../build
```

- Then go to the build directory and call make for compilation: 
```
make 
```

## Quick Start

- The hello world example for this package can be found in the folder: Tests/Optimization/Himmelblau/Unconstrained
- Change directory to this folder and run the python script runOptimization.py: 
```
python runOptimization.py 
```


