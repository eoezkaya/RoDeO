#!/usr/bin/env python3

import numpy as np 
from mishra05_optimization import Mishra05Optimization

func = Mishra05Optimization()

print("Evaluating the Mishra05 function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = func.evaluateFunction(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
