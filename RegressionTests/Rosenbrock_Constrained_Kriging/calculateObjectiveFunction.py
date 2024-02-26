#!/usr/bin/env python3

import numpy as np 

def Rosenbrock(x):
    return (1-x[0])**2 + 100.0 * (x[1] - x[0]*x[0])**2	
    
    


print("Evaulating the Rosenbrock function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = Rosenbrock(dv)   

print('function value = ', functionValue)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.close()
