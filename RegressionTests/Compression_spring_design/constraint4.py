#!/usr/bin/env python3
import numpy as np

def h4(x):
    return (x[0]+x[1])/1.5 - 1.0
    
    
print("Evaluating the constraint function...\n")

dim = 3
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = h4(dv)   

print('h3(x) = ', constraintValue)


f = open("constraint4.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


