import os
import numpy as np
import pandas as pd 
import alpine02_5D as alp



RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "alpine02_5D.cfg"


NTrainingSamples = 100

Xsamples = np.random.rand(NTrainingSamples,5)*10

print("Number of samples used in the DoE = \n",NTrainingSamples)
print("DoE samples = \n", Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,11)

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
	fValue = alp.Alpine02_5D(xp)
	ObjFunSamples[i,5] = fValue
	if(fValue<bestValue):
	    bestValue = fValue
	    bestValueIndex = i
	    bestx = xp
	
	ObjFunSamples[i,6] = 0.0
	ObjFunSamples[i,7] = 0.0
	ObjFunSamples[i,8] = 0.0
	ObjFunSamples[i,9] = 0.0
	ObjFunSamples[i,10] = 0.0

grad = alp.Alpine02_5D_Gradient(bestx)
ObjFunSamples[bestValueIndex,6] = grad[0]
ObjFunSamples[bestValueIndex,7] = grad[1]
ObjFunSamples[bestValueIndex,8] = grad[2]
ObjFunSamples[bestValueIndex,9] = grad[3]
ObjFunSamples[bestValueIndex,10] = grad[4]


df = pd.DataFrame(ObjFunSamples)
df.to_csv("alpine02_5D.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)





