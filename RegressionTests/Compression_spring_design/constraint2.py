#!/usr/bin/env python3
import numpy as np

def h2(x):
    return (4.0*x[1]**2-x[0]*x[1])/(12566*x[1]*x[0]**3 - x[0]**4) + 1.0/(5108*x[0]**2) - 1.0
    
    
print("Evaluating the constraint function...\n")

dim = 3
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = h2(dv)   

print('h2(x) = ', constraintValue)


f = open("constraint2.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


