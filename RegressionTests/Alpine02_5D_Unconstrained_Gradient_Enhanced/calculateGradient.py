#!/usr/bin/env python3

import numpy as np 

def Alpine02_5D(x):
    prod = 1.0
    for i in range(5):
        prod = prod * np.sqrt(x[i])*np.sin(x[i])
    return prod


def Alpine02_5D_Gradient(x):
    print(x)
    grad = np.zeros(5)
    f0 = Alpine02_5D(x)
    for i in range(5):
        epsilon = x[i]*0.001
        x[i] = x[i] + epsilon
        fp = Alpine02_5D(x)
        x[i] = x[i] - epsilon
        grad[i] = (fp - f0)/epsilon
    return grad        
        
    


print("Evaulating the gradient of the Alpine02 function (5D) ...\n")

dim = 5
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
grad = Alpine02_5D_Gradient(dv)   

print('gradient vector = ', grad)


f = open("gradientVector.dat", "w")
for i in range(5):
    f.write(str(grad[i]) + "\n")
f.close()
