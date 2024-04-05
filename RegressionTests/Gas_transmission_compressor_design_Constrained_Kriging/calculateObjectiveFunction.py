#!/usr/bin/env python3

import numpy as np 

def func(x):
    return 8.61*10**5*x[0]**0.5*x[1]*x[2]**(-2/3)*x[3]**(-0.5) + 3.69*10**4*x[2] + 7.72*10**8*(1.0/x[0])*x[1]**0.219 - 765.43*10**6*(1.0/x[0]) 	
    
    


print("Evaulating the objective function...\n")

dim = 4
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
