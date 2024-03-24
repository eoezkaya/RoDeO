#!/usr/bin/env python3
import numpy as np
from welded_beam_optimization import WeldedBeamOptimization

    
function = WeldedBeamOptimization()
    
print("Evaulating the constraint function...\n")

dim = 4
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = function.evaluateConstraint2(dv)   

print('function value = ', constraintValue)


f = open("constraint2.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


