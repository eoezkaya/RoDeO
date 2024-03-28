import os
import numpy as np
import pandas as pd 

from paviani_optimization import PavianiOptimization

    
    
ub = [9.999, 9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999]
lb = [2.001, 2.001,2.001,2.001,2.001,2.001,2.001,2.001,2.001,2.001]
func = PavianiOptimization()
NTrainingSamples = 100
RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "paviani.cfg"


print("Number of samples used in the DoE = \n",NTrainingSamples)

ObjFunSamples = np.random.rand(NTrainingSamples,11)

for i in range(NTrainingSamples):
    x = np.random.rand(10)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    x[2] = x[2]*(ub[2]-lb[2])+lb[2]
    x[3] = x[3]*(ub[3]-lb[3])+lb[3]
    x[4] = x[4]*(ub[4]-lb[4])+lb[4]
    x[5] = x[5]*(ub[5]-lb[5])+lb[5]
    x[6] = x[6]*(ub[6]-lb[6])+lb[6]
    x[7] = x[7]*(ub[7]-lb[7])+lb[7]
    x[8] = x[8]*(ub[8]-lb[8])+lb[8]
    x[9] = x[9]*(ub[9]-lb[9])+lb[9]
    ObjFunSamples[i,0] = x[0]
    ObjFunSamples[i,1] = x[1]
    ObjFunSamples[i,2] = x[2]
    ObjFunSamples[i,3] = x[3]
    ObjFunSamples[i,4] = x[4]
    ObjFunSamples[i,5] = x[5]
    ObjFunSamples[i,6] = x[6]
    ObjFunSamples[i,7] = x[7]
    ObjFunSamples[i,8] = x[8]
    ObjFunSamples[i,9] = x[9]
    ObjFunSamples[i,10] = func.evaluateFunction(x)	
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("paviani.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



