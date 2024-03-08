import os
import numpy as np

def func(x):    
    return x[0]**2.0*x[1]*(2.0+x[2])

def h1(x):
    return 1 - (x[1]**3*x[2])/(71785*x[0]**4)
def h2(x):
    return (4.0*x[1]**2-x[0]*x[1])/(12566*x[1]*x[0]**3 - x[0]**4) + 1.0/(5108*x[0]**2) - 1.0
def h3(x):
    return 1.0- (140.45*x[0])/(x[1]**2*x[2])
def h4(x):
    return (x[0]+x[1])/1.5 - 1.0


RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "compression_spring_design.cfg"

NTrainingSamples = 50


ObjFunSamples = np.random.rand(NTrainingSamples,4)
Constraint1Samples = np.random.rand(NTrainingSamples,4)
Constraint2Samples = np.random.rand(NTrainingSamples,4)
Constraint3Samples = np.random.rand(NTrainingSamples,4)
Constraint4Samples = np.random.rand(NTrainingSamples,4)

for i in range(NTrainingSamples):
    x = np.random.rand(3)
    x[0] = x[0]*(2-0.05)+0.05
    x[1] = x[1]*(1.3-0.25)+0.25
    x[2] = x[2]*(15-2)+2
    f = func(x)	
    c1 = h1(x)
    c2 = h2(x)
    c3 = h3(x)
    c4 = h4(x)
    ObjFunSamples[i,0] = x[0]
    ObjFunSamples[i,1] = x[1]
    ObjFunSamples[i,2] = x[2]
    ObjFunSamples[i,3] = f
    Constraint1Samples[i,0] = x[0]
    Constraint1Samples[i,1] = x[1]
    Constraint1Samples[i,2] = x[2]
    Constraint1Samples[i,3] = c1
    Constraint2Samples[i,0] = x[0]
    Constraint2Samples[i,1] = x[1]
    Constraint2Samples[i,2] = x[2]
    Constraint2Samples[i,3] = c2
    Constraint3Samples[i,0] = x[0]
    Constraint3Samples[i,1] = x[1]
    Constraint3Samples[i,2] = x[2]
    Constraint3Samples[i,3] = c3
    Constraint4Samples[i,0] = x[0]
    Constraint4Samples[i,1] = x[1]
    Constraint4Samples[i,2] = x[2]
    Constraint4Samples[i,3] = c4
	

import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("compression_spring_design.csv",header=False, index=False)
df = pd.DataFrame(Constraint1Samples)
df.to_csv("constraint1.csv",header=False, index=False)
df = pd.DataFrame(Constraint2Samples)
df.to_csv("constraint2.csv",header=False, index=False)
df = pd.DataFrame(Constraint3Samples)
df.to_csv("constraint3.csv",header=False, index=False)
df = pd.DataFrame(Constraint4Samples)
df.to_csv("constraint4.csv",header=False, index=False)



COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

