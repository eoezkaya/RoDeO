# RoDeO
RoDeo (Robust Design Optimization Package) is a package for simulation based global design optimization. It is specifically designed
for scientific/engineering applications, in which the objective function and constraints are evaluated by computationally expensive simulations. 

## Features of the RoDeO Package:
- Surrogate model based Efficient Global Optimization (EGO) strategy
- Full data-driven approach
- Easy and efficient treatment of inequality constraints

 
## External Libraries
 
 
## Build Support

- RoDeO uses cmake utility for the compilation and the build process. 
- First change directory to **src** and call **cmake**:
```
cmake -B ../build
```

- Then go to the **build** directory and call make for compilation: 
```
make 
```

## Quick Start

- The hello world example for this package can be found in the folder:  
**Tests/Rosenbrock_Constrained**
- Change directory to this folder and run the python script **runOptimization.py**: 
```
python runOptimization.py 
```


