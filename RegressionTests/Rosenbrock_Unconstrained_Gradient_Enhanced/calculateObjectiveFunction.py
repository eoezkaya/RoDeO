#!/usr/bin/env python3

import numpy as np 
from rosenbrock_optimization import RosenbrockOptimization

    
rosenbrockOpt = RosenbrockOptimization()

print("Evaulating the Rosenbrock function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = rosenbrockOpt.evaluateFunction(dv) 

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
