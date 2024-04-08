#!/usr/bin/env python3

import numpy as np 
from michalewitz_optimization import MichalewitzOptimization

func = MichalewitzOptimization()

print("Evaluating the Michalewitz function...\n")

dim = 10
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
