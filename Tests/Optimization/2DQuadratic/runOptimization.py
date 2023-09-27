import os
import numpy as np


BIN_RODEO = "../../../build/rodeo"
configFilename = "2DQuadratic.cfg"
# compile the code that evaluates the objective function
os.system("g++ 2DQuadratic.cpp -o 2DQuadratic -larmadillo")
os.system("g++ constraint1.cpp -o constraint1")

NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,2)
Xsamples = Xsamples * 60 - 30
ObjFunSamples = np.random.rand(NTrainingSamples,3)
Constraint1Samples = np.random.rand(NTrainingSamples,3)

for i in range(NTrainingSamples):
	xp = Xsamples[i,0]
	yp = Xsamples[i,1]
	
	ObjFunSamples[i,0] = xp
	ObjFunSamples[i,1] = yp
	ObjFunSamples[i,2] = 0.5*xp**2 + yp**2 - xp*yp + 2*xp + 6*yp
	Constraint1Samples[i,0] = xp
	Constraint1Samples[i,1] = yp
	Constraint1Samples[i,2] = xp + yp


import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("2DQuadratic.csv",header=False, index=False)
df = pd.DataFrame(Constraint1Samples)
df.to_csv("constraint1.csv",header=False, index=False)


COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

