#!/usr/bin/env python3
import numpy as np
def evaluateConstraint(x):
    return (x[0] - 1)**3 - x[1] + 1;
    
    
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


f = open("constraint2.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


