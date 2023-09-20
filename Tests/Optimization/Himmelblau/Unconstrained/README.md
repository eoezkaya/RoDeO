## Description

In this example, the Himmelblau test function is minimized without any external constraint. We have only box constraints for the parameters. The optimization problem is given as

$minimize$ $f(x_1,x_2) = (x_1^2+x_2-11.0)^2 + (x_1+x_2^2-7.0)^2$ 

$subject$ $to$  $-6.0 < x_1 < 6.0$ and $-6.0 < x_2 < 6.0$.



The optimization problem has four identical local solutions:

$f(3.0,2.0) = 0.0$

$f(-2.805118, 3.131312) = 0.0$

$f(-3.779319, -3.283186) = 0.0$

$f(3.584428, -1.848126) = 0.0$


The Himmelblau function is implemented in the c-file "himmelblau.cpp". This small program reads the values of $x_1$ and $x_2$ from the file "dv.dat" and writes the output to 
the file "objFunVal.dat". 


## Running the test case

- To run the test case, run the python file "runOptimization.py" in terminal:

```
python runOptimization.py 
```

