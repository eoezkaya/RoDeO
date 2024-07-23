## Description

In this example, the 2D Rosenbrock function is minimized with two external non-linear constraints. The optimization problem is given as

```math
\text{minimize  } f(x_1,x_2) = (1-x_1)^2 + 100(x_2-x_1^2)^2
```
```math
\text{subject to  } x_2 - x_1^2 > 0, (x_1-1)^3-x_2+0.7 > 0, -1.5 \leq x_1 \leq 1.5  \text{ and } -0.5 \leq x_2 \leq 2.5.
```
The contour plot of the function is displayed below:

<img src="./rosenbrock.png" alt="Rosenbrock function" title="Rosenbrock function">



## Running the test case

- To run the test case, run the python file "runOptimization.py" in terminal:

```
python runOptimization.py 
```




