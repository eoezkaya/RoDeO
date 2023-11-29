import os
import numpy as np
import pandas as pd 

RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "himmelblau.cfg"
# compile the code that evaluates the objective function
os.system("g++ himmelblau.cpp -o himmelblau")


NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,2)
Xsamples[:,0] = Xsamples[:,0] * 12 - 6.0
Xsamples[:,1] = Xsamples[:,1] * 12 - 6.0

print("Number of samples used in the DoE = \n",NTrainingSamples)
print("DoE samples = \n", Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,5)

bestValue = 1000000
bestValueIndex = 0
for i in range(NTrainingSamples):
	xp = Xsamples[i,0]
	yp = Xsamples[i,1]
	
	ObjFunSamples[i,0] = xp
	ObjFunSamples[i,1] = yp
	fValue = (((xp**2+yp-11)**2) + (((xp+yp**2-7)**2)))
	ObjFunSamples[i,2] = fValue
	if(fValue<bestValue):
	    bestValue = fValue
	    bestValueIndex = i
	
	ObjFunSamples[i,3] = 0.0
	ObjFunSamples[i,4] = 0.0

xp = Xsamples[bestValueIndex,0]
yp = Xsamples[bestValueIndex,1]
ObjFunSamples[bestValueIndex,3] = 4*xp*(xp**2+yp-11) + 2*(xp+yp**2-7)
ObjFunSamples[bestValueIndex,4] = (xp**2+yp-11) + 4*yp*(xp+yp**2-7)



df = pd.DataFrame(ObjFunSamples)
df.to_csv("himmelblau.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)





