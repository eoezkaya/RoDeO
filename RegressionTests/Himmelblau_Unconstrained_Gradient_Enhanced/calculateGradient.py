#!/usr/bin/env python3

import numpy as np 
from himmelblau_optimization import HimmelblauOptimization
himmelblauOpt = HimmelblauOptimization()



print("Evaluating the gradient of the Himmelblau function...\n")

dim = 2
dv = np.zeros(dim)
            
f = open('dv.dat', "r")
for i in range(dim):
    line = f.readline()
    dv[i] = float(line)
f.close()            

print('design variables = ',dv)
            
gradient  = himmelblauOpt.evaluateGradient(dv)


print('gradient = ', gradient)

f = open("gradientVector.dat", "w")
f.write(str(gradient[0]) + "\n")
f.write(str(gradient[1]) + "\n")
f.close()
