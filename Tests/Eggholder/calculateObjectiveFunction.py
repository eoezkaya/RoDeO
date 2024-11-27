#!/usr/bin/env python3

import numpy as np 
from eggholder_optimization import EggholderOptimization
optimizer = EggholderOptimization()


print("Evaluating the Eggholder function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = optimizer.evaluateFunction(dv)   

print('function value = ', functionValue)


f = open("objective.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
