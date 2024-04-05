import os
import numpy as np
import pandas as pd 

from rastrigin_optimization import RastriginOptimization


ub = 15*[4.12]
lb = 15*[-7.12]

func = RastriginOptimization()
NTrainingSamples = 200
RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "rastrigin.cfg"


print("Number of samples used in the DoE = \n",NTrainingSamples)

ObjFunSamples = np.random.rand(NTrainingSamples,16)

for i in range(NTrainingSamples):
    x = np.random.rand(15)
    for j in range(15):
        x[j] = x[j]*(ub[j]-lb[j])+lb[j]
    for j in range(15):
        ObjFunSamples[i,j] = x[j]    
    ObjFunSamples[i,15] = func.evaluateFunction(x)	
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("rastrigin.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



