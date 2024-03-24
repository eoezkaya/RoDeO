#!/usr/bin/env python3

import numpy as np 
from welded_beam_optimization import WeldedBeamOptimization

    
function = WeldedBeamOptimization()

print("Evaulating the objective function for welded beam problem...\n")

dim = 4
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = function.evaluateFunction(dv) 

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
