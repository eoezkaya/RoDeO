import os
import numpy as np

from welded_beam_optimization import WeldedBeamOptimization

    
    
ub = [2.0, 10.0, 10.0, 2.0]
lb = [0.125, 0.1, 0.1, 0.1 ]

func = WeldedBeamOptimization()

RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "welded_beam.cfg"

NTrainingSamples = 50
dim = 4

ObjFunSamples = np.random.rand(NTrainingSamples,dim+1)
Constraint1Samples = np.random.rand(NTrainingSamples,dim+1)
Constraint2Samples = np.random.rand(NTrainingSamples,dim+1)
Constraint3Samples = np.random.rand(NTrainingSamples,dim+1)
Constraint4Samples = np.random.rand(NTrainingSamples,dim+1)
Constraint5Samples = np.random.rand(NTrainingSamples,dim+1)

for i in range(NTrainingSamples):

    x = np.random.rand(dim)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    x[2] = x[2]*(ub[2]-lb[2])+lb[2]
    x[3] = x[3]*(ub[3]-lb[3])+lb[3]
    ObjFunSamples[i,0] = x[0]
    ObjFunSamples[i,1] = x[1]
    ObjFunSamples[i,2] = x[2]
    ObjFunSamples[i,3] = x[3]
    
    ObjFunSamples[i,4] = func.evaluateFunction(x)
   
        
        
    Constraint1Samples[i,0] = x[0]
    Constraint1Samples[i,1] = x[1]
    Constraint1Samples[i,2] = x[2]
    Constraint1Samples[i,3] = x[3]
    Constraint1Samples[i,4] = func.evaluateConstraint1(x)
    
    
    Constraint2Samples[i,0] = x[0]
    Constraint2Samples[i,1] = x[1]
    Constraint2Samples[i,2] = x[2]
    Constraint2Samples[i,3] = x[3]
    Constraint2Samples[i,4] = func.evaluateConstraint2(x)

    Constraint3Samples[i,0] = x[0]
    Constraint3Samples[i,1] = x[1]
    Constraint3Samples[i,2] = x[2]
    Constraint3Samples[i,3] = x[3]
    Constraint3Samples[i,4] = func.evaluateConstraint3(x)

    Constraint4Samples[i,0] = x[0]
    Constraint4Samples[i,1] = x[1]
    Constraint4Samples[i,2] = x[2]
    Constraint4Samples[i,3] = x[3]
    Constraint4Samples[i,4] = func.evaluateConstraint4(x)

    Constraint5Samples[i,0] = x[0]
    Constraint5Samples[i,1] = x[1]
    Constraint5Samples[i,2] = x[2]
    Constraint5Samples[i,3] = x[3]
    Constraint5Samples[i,4] = func.evaluateConstraint5(x)



import pandas as pd 
df = pd.DataFrame(ObjFunSamples)
df.to_csv("welded_beam.csv",header=False, index=False)
df = pd.DataFrame(Constraint1Samples)
df.to_csv("constraint1.csv",header=False, index=False)
df = pd.DataFrame(Constraint2Samples)
df.to_csv("constraint2.csv",header=False, index=False)
df = pd.DataFrame(Constraint3Samples)
df.to_csv("constraint3.csv",header=False, index=False)
df = pd.DataFrame(Constraint4Samples)
df.to_csv("constraint4.csv",header=False, index=False)
df = pd.DataFrame(Constraint5Samples)
df.to_csv("constraint5.csv",header=False, index=False)
COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

