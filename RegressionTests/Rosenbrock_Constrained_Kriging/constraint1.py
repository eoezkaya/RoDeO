#!/usr/bin/env python3
import numpy as np

def evaluateConstraint(x):
    return x[1] - x[0]*x[0]
    
    
print("Evaulating the constraint function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = evaluateConstraint(dv)   

print('function value = ', constraintValue)


f = open("constraint1.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


