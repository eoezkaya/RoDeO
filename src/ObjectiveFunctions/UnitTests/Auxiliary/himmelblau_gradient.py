#!/usr/bin/env python3

import numpy as np 

def evaluateGradient(x):
        temp1 = (x[0]**2+x[1]-11)
        temp2 = (x[0]+x[1]**2-7)
        dfdx1 = 4*temp1*x[0] + 2*temp2
        dfdx2 = 2*temp1      + 4*temp2*x[1]
        return [dfdx1,dfdx2]    




#print("Evaluating the gradient of the Himmelblau function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

#print('design variables = ',dv)
            

gradient      = evaluateGradient(dv)

#print('gradient = ', gradient)

f = open("gradientVector.dat", "w")
f.write(str(gradient[0]) + "\n")
f.write(str(gradient[1]) + "\n")
f.close()
