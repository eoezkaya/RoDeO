#!/usr/bin/env python3

import numpy as np 


def degToRad(degree):
    return degree / 180.0 * np.pi
def Wingweight(x):
    return 0.036* x[0]**0.758* x[1]**0.0035* (x[2] / np.cos(degToRad(x[3])) ** 2) ** (0.6)* x[4]**0.006* x[5]**0.04* (100 * x[6] / np.cos(degToRad(x[3]))) ** (-0.3)* (x[7] * x[8]) ** 0.49+ x[0] * x[9]
    

print("Evaulating the Wingweight function...\n")

dim = 10
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = Wingweight(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
