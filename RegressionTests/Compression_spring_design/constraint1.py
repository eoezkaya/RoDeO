#!/usr/bin/env python3
import numpy as np

def h1(x):
    return 1 - (x[1]**3*x[2])/(71785*x[0]**4)
    
    
print("Evaluating the constraint function...\n")

dim = 3
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = h1(dv)   

print('h1(x) = ', constraintValue)


f = open("constraint1.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


