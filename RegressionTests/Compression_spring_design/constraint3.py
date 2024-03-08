#!/usr/bin/env python3
import numpy as np

def h3(x):
    return 1.0- (140.45*x[0])/(x[1]**2*x[2])
    
    
print("Evaluating the constraint function...\n")

dim = 3
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
constraintValue = h3(dv)   

print('h3(x) = ', constraintValue)


f = open("constraint3.dat", "w")
f.write(str(constraintValue) + "\n")
f.close()


