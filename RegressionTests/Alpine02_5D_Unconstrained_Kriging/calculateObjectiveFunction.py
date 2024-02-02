#!/usr/bin/env python3

import numpy as np 

def Alpine02_5D(x):
    print(x)
    prod = 1.0
    for i in range(5):
        prod = prod * np.sqrt(x[i])*np.sin(x[i])
    return prod
    


print("Evaulating the Alpine02 function (5D) ...\n")

dim = 5
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = Alpine02_5D(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
