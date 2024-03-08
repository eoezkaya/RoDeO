#!/usr/bin/env python3

import numpy as np 

def func(x):    
    return x[0]**2.0*x[1]*(2.0+x[2])


print("Evaulating the objective function...\n")

dim = 3
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = func(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
