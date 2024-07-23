#!/usr/bin/env python3

import numpy as np 

def evaluateFunction(x):
    return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))	     




dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

#print('design variables = ',dv)
            

functionValue      =  evaluateFunction(dv)

#print('gradient = ', gradient)

f = open("objectiveFunction.dat", "w")
f.write(str(functionValue ) + "\n")
f.close()
