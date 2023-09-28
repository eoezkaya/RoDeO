## Description

In this example, the Wingweight test function is minimized without any constraint. The Wingweight
function is given as

```math
W = 0.036S_w^0.758 W_{fw}^0.0035 * (A/(cos(deg)*cos(deg)))^0.6 * q^0.006 * lambda^ 0.04 * (100.0*tc/cos(deg))^{-0.3}) (N_z*W_dg)^0.49 + S_w*W_p
```

The parameters of the function are:


- $S_w$: Wing Area (ft^2) (150,200)
- Wfw: Weight of fuel in the wing (lb) (220,300)
- A: Aspect ratio (6,10)
- $/Lambda$: quarter chord sweep (deg) (-10,10)
- q: dynamic pressure at cruise (lb/ft^2)  (16,45)
- $/lambda$: taper ratio (0.5,1)
- $t_c$: aerofoil thickness to chord ratio (0.08,0.18)
- $N_z$: ultimate load factor (2.5,6)
- $W_{dg}$: flight design gross weight (lb)  (1700,2500)
- $W_p$ : paint weight (lb/ft^2) (0.025, 0.08)


Brute Force Global Optimization Results:
Function has minimum at x:
   1.5008e+02   2.5878e+02   6.0596e+00   8.1827e-01   1.6641e+01   8.8329e-01   1.7650e-01   2.5030e+00   1.7174e+03   2.6899e-02
Function value at maximum = 128.5995668
Function has maximum at x:
   1.9778e+02   2.8262e+02   9.9356e+00  -7.8339e+00   4.4301e+01   9.9878e-01   8.0261e-02   5.9803e+00   2.4768e+03   2.7524e-02
Function value at maximum = 494.6254512




To run the test case, run the python file "runOptimization.py" in terminal: python runOptimization.py 

## Running the test case

- To run the test case, run the python file "runOptimization.py" in terminal:

```
python runOptimization.py 
```


## Visualizing the results 

- To visualize the samples generated in the optimization process, run the python file "plotResults.py" in terminal:

```
python plotResults.py 
```

<img src="./rosenbrockResults.png" alt="Rosenbrock function" title="Rosenbrock function">

