#!/usr/bin/env python3

import numpy as np 

from rosenbrock_optimization import RosenbrockOptimization

rosenbrockOpt = RosenbrockOptimization()

print("Evaluating the gradient of the Rosenbrock function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
gradient = rosenbrockOpt.evaluateGradient(dv)

print('gradient = ', gradient)

f = open("gradientVector.dat", "w")
f.write(str(gradient[0]) + "\n")
f.write(str(gradient[1]) + "\n")
f.close()
