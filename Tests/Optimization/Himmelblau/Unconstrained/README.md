## Description

In this example, the Himmelblau test function is minimized without any constraint. The optimization problem is given as

$minimize f(x1,x2) = (x1^2+x2-11.0)^2 + (x1+x2^2-7.0)^2$.
$subject to -6.0 < x1 < 6.0 and -6.0 < x2 < 6.0$



It has four identical local minima:

$f(3.0,2.0) = 0.0$
$f(-2.805118, 3.131312) = 0.0$
$f(-3.779319, -3.283186) = 0.0$
$f(3.584428, -1.848126) = 0.0$


The Himmelblau function is implemented in the c-file "himmelblau.cpp" and writes the output to 
a file "objFunVal.dat". 


## Running the test case

- To run the test case, run the python file "runOptimization.py" in terminal:

```
python runOptimization.py 
```

