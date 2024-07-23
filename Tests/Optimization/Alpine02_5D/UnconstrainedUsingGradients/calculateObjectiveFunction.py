#!/usr/bin/env python3

import numpy as np 
import alpine02_5D as alp

print("\nEvaluating the Alpine02 function (5 input variables) with gradients...\n")

dim = 5
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
functionValue = alp.Alpine02_5D(dv)   

print('function value = ', functionValue)


fd = alp.Alpine02_5D_Gradient(dv)

print('gradient vector = ', fd)


f = open("objectiveFunction.dat", "w")
f.write(str(functionValue) + "\n")
f.write(str(fd[0])+ "\n")
f.write(str(fd[1])+ "\n")
f.write(str(fd[2])+ "\n")
f.write(str(fd[3])+ "\n")
f.write(str(fd[4]))
f.close()
