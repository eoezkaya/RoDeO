#!/usr/bin/env python3

import numpy as np 

def Eggholder(x):
    return -(x[1]+47.0)*np.sin(np.sqrt(abs(x[1]+0.5*x[0]+47.0)))-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47.0) )))	
    

print("Evaulating the Eggholder function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = Eggholder(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()




