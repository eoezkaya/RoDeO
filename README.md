# RoDeO
RoDeo (Robust Design Optimization Package) is a package for simulation based global design optimization. It is specifically designed
for scientific/engineering applications, in which the objective function and constraints are evaluated by computationally expensive simulations. 

Features of the RoDeO Package:
- Surrogate model based Efficient Global Optimization (EGO) strategy
- Full data-driven approach
- Easy and efficient treatment of inequality constraints
- Gradient and tangent enhaced surrogate modeling 
 
 
External Libraries
 - RoDeO uses the linear algebra library Armadillo. Therefore, before compiling RoDeO sources, Armadillo must be installed in the system. To download Armadillo and
 see the documentation please refer to https://arma.sourceforge.net/ 
 
Build Support

- RoDeO uses cmake utility for the compilation and the build process. 
- First go to the src directory and call cmake:
-- cmake -B ../build/
