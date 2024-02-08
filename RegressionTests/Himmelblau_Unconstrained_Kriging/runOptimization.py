import os
import numpy as np
import pandas as pd 

RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "himmelblau.cfg"

NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,2)
Xsamples[:,0] = Xsamples[:,0] * 12 - 6.0
Xsamples[:,1] = Xsamples[:,1] * 12 - 6.0

print("Number of samples used in the DoE = \n",NTrainingSamples)
print("DoE samples = \n", Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,3)

for i in range(NTrainingSamples):
	xp = Xsamples[i,0]
	yp = Xsamples[i,1]
	
	ObjFunSamples[i,0] = xp
	ObjFunSamples[i,1] = yp
	ObjFunSamples[i,2] = (((xp**2+yp-11)**2) + (((xp+yp**2-7)**2)))
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("himmelblau.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



