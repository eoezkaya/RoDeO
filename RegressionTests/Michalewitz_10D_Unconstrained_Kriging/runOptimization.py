import os
import numpy as np
import pandas as pd 

from michalewitz_optimization import MichalewitzOptimization


ub = 10*[3.141592]
lb = 10*[0]

func = MichalewitzOptimization()
NTrainingSamples = 150
RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "michalewitz.cfg"


print("Number of samples used in the DoE = \n",NTrainingSamples)

ObjFunSamples = np.random.rand(NTrainingSamples,11)

for i in range(NTrainingSamples):
    x = np.random.rand(10)
    for j in range(10):
        x[j] = x[j]*(ub[j]-lb[j])+lb[j]
    for j in range(10):
        ObjFunSamples[i,j] = x[j]    
    ObjFunSamples[i,10] = func.evaluateFunction(x)	
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("michalewitz.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



