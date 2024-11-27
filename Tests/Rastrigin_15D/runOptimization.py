import os
import numpy as np
import pandas as pd 
import glob

from rastrigin_optimization import RastriginOptimization


# Remove all .log files in the current directory
for log_file in glob.glob("*.log"):
    try:
        os.remove(log_file)
        print(f"Removed log file: {log_file}")
    except OSError as e:
        print(f"Error removing file {log_file}: {e}")

ub = 15*[4.12]
lb = 15*[-7.12]

func = RastriginOptimization()
NTrainingSamples = 200
RODEO_HOME = "/home/eoezkaya/RoDOP"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "rastrigin.xml"


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



