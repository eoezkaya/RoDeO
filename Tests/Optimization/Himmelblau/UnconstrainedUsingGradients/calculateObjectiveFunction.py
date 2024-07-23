#!/usr/bin/env python3

import numpy as np 

def Himmelblau(x):
    return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))

def HimmelblauGradient(x):
    temp1 = (x[0]**2+x[1]-11)
    temp2 = (x[0]+x[1]**2-7)
    dfdx1 = 4*temp1*x[0] + 2*temp2
    dfdx2 = 2*temp1      + 4*temp2*x[1]
    
    return [dfdx1,dfdx2]


print("Evaulating the Himmelblau function with gradients...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = Himmelblau(dv)   

print('funtion value = ', functionValue)


[dfdx1, dfdx2] = HimmelblauGradient(dv)

print('gradient vector = ', dfdx1, dfdx2)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.write(str(dfdx1)+ "\n")
f.write(str(dfdx2))
f.close()
