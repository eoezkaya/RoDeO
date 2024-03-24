import os
import numpy as np
import pandas as pd 


def Alpine02_5D(x):
    prod = 1.0
    for i in range(5):
        prod = prod * np.sqrt(x[i])*np.sin(x[i])
    return prod



RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "alpine02_5D.cfg"


NTrainingSamples = 100

Xsamples = np.random.rand(NTrainingSamples,5)*10

print("Number of samples used in the DoE = \n",NTrainingSamples)
print("DoE samples = \n", Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,6)

bestValue = 1000000
bestValueIndex = 0
bestx = np.zeros(5)
for i in range(NTrainingSamples):
	x1 = Xsamples[i,0]
	x2 = Xsamples[i,1]
	x3 = Xsamples[i,2]
	x4 = Xsamples[i,3]
	x5 = Xsamples[i,4]
	
	ObjFunSamples[i,0] = x1
	ObjFunSamples[i,1] = x2
	ObjFunSamples[i,2] = x3
	ObjFunSamples[i,3] = x4
	ObjFunSamples[i,4] = x5
	
	xp = [x1,x2,x3,x4,x5]
	fValue = Alpine02_5D(xp)
	ObjFunSamples[i,5] = fValue
	
df = pd.DataFrame(ObjFunSamples)
df.to_csv("alpine02_5D.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)





