#!/usr/bin/env python3

import numpy as np 

from rosenbrock_optimization import RosenbrockOptimization

rosenbrockOpt = RosenbrockOptimization()

print("Evaluating the Rosenbrock function with gradient...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = rosenbrockOpt.evaluateFunction(dv)   
gradient = rosenbrockOpt.evaluateGradient(dv)

print('function value = ', functionValue)
print('gradient = ', gradient)

f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.write(str(gradient[0]) + "\n")
f.write(str(gradient[1]) + "\n")
f.close()
