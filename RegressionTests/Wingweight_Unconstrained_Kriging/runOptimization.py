import os
import numpy as np

def degToRad(degree):
    return degree / 180.0 * np.pi
def Wingweight(x):
    return 0.036* x[0]**0.758* x[1]**0.0035* (x[2] / np.cos(degToRad(x[3])) ** 2) ** (0.6)* x[4]**0.006* x[5]**0.04* (100 * x[6] / np.cos(degToRad(x[3]))) ** (-0.3)* (x[7] * x[8]) ** 0.49+ x[0] * x[9]
    

RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "wingweight.cfg"

NTrainingSamples = 50

Xsamples = np.random.rand(NTrainingSamples,10)
Xsamples[:,0] = Xsamples[:,0] * 50 + 150
Xsamples[:,1] = Xsamples[:,1] * 80 + 220
Xsamples[:,2] = Xsamples[:,2] * 4 + 6.0
Xsamples[:,3] = Xsamples[:,3] * 20 - 10.0
Xsamples[:,4] = Xsamples[:,4] * 29 + 16.0
Xsamples[:,5] = Xsamples[:,5] * 0.5 + 0.5
Xsamples[:,6] = Xsamples[:,6] * 0.1 + 0.08
Xsamples[:,7] = Xsamples[:,7] * 3.5 + 2.5
Xsamples[:,8] = Xsamples[:,8] * 800 + 1700
Xsamples[:,9] = Xsamples[:,9] * 0.055 + 0.025



print(Xsamples)

ObjFunSamples = np.random.rand(NTrainingSamples,11)


for i in range(NTrainingSamples):
	
	ObjFunSamples[i,0:10] = Xsamples[i,0:10]
	ObjFunSamples[i,10] = Wingweight(Xsamples[i])
	

import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("wingweight.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

