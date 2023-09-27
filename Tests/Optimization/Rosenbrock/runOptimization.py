import os
import numpy as np


BIN_RODEO = "../../../build/rodeo"
configFilename = "Rosenbrock.cfg"
# compile the code that evaluates the objective function
os.system("g++ Rosenbrock.cpp -o Rosenbrock -lm")
os.system("g++ constraint1.cpp -o constraint1")
os.system("g++ constraint2.cpp -o constraint2")

NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,2)
Xsamples[:,0] = Xsamples[:,0] * 3 - 1.5
Xsamples[:,1] = Xsamples[:,1] * 3 - 0.5

print(Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,3)
Constraint1Samples = np.random.rand(NTrainingSamples,3)
Constraint2Samples = np.random.rand(NTrainingSamples,3)

for i in range(NTrainingSamples):
	xp = Xsamples[i,0]
	yp = Xsamples[i,1]
	
	ObjFunSamples[i,0] = xp
	ObjFunSamples[i,1] = yp
	ObjFunSamples[i,2] = (1-xp)**2 + 100*(yp-xp**2)**2
	Constraint1Samples[i,0] = xp
	Constraint1Samples[i,1] = yp
	Constraint1Samples[i,2] = xp + yp
	Constraint2Samples[i,0] = xp
	Constraint2Samples[i,1] = yp	
	Constraint2Samples[i,2] = (xp-1)**3 + yp +1


import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("Rosenbrock.csv",header=False, index=False)
df = pd.DataFrame(Constraint1Samples)
df.to_csv("constraint1.csv",header=False, index=False)
df = pd.DataFrame(Constraint2Samples)
df.to_csv("constraint2.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

