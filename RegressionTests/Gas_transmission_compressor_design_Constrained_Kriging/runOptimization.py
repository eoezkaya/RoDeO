import os
import numpy as np


RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "gas_transmission.cfg"

NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,4)
Xsamples[:,0] = Xsamples[:,0] * 30 + 20.0
Xsamples[:,1] = Xsamples[:,1] * 9  + 1
Xsamples[:,2] = Xsamples[:,2] * 30 + 20.0
Xsamples[:,3] = Xsamples[:,3] * 59.9 + 0.1

print(Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,5)
Constraint1Samples = np.random.rand(NTrainingSamples,5)


for i in range(NTrainingSamples):
	x = Xsamples[i,:]	
	ObjFunSamples[i,0] = x[0]
	ObjFunSamples[i,1] = x[1]
	ObjFunSamples[i,2] = x[2]
	ObjFunSamples[i,3] = x[3]	
	ObjFunSamples[i,4] = 8.61*10**5*x[0]**0.5*x[1]*x[2]**(-2/3)*x[3]**(-0.5) + 3.69*10**4*x[2] + 7.72*10**8*(1.0/x[0])*x[1]**0.219 - 765.43*10**6*(1.0/x[0]) 	
	
	Constraint1Samples[i,0] = x[0]
	Constraint1Samples[i,1] = x[1]
	Constraint1Samples[i,2] = x[2]
	Constraint1Samples[i,3] = x[3]
	Constraint1Samples[i,4] = x[3]*x[1]**-2 + x[1]**-2 - 1
	


import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("gas_transmission.csv",header=False, index=False)
df = pd.DataFrame(Constraint1Samples)
df.to_csv("constraint1.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

