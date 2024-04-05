#!/usr/bin/env python3
import numpy as np

def h1(x):
    return x[3]*x[1]**-2 + x[1]**-2 - 1
    
    
print("Evaulating the constraint function...\n")

dim = 4
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


