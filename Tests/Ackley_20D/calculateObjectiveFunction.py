#!/usr/bin/env python3

import numpy as np 
from ackley_optimization import AckleyOptimization

func = AckleyOptimization()

print("Evaluating the Ackley function (20D)...\n")

dim = 20
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = func.evaluateFunction(dv)   

print('function value = ', functionValue)


f = open("objective.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
