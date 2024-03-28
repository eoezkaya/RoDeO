import os
import numpy as np

from damavandi_optimization import DamavandiOptimization

    
func = DamavandiOptimization()

RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "damavandi.cfg"

NTrainingSamples = 50

ub = [14.0, 14.0]
lb = [0.0, 0.0]


ObjFunSamples = np.random.rand(NTrainingSamples,3)

for i in range(NTrainingSamples):

    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    ObjFunSamples[i,0] = x[0]
    ObjFunSamples[i,1] = x[1]
    ObjFunSamples[i,2] = func.evaluateFunction(x)
   


import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("damavandi.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

