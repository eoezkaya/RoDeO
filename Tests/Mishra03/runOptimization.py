import os
import numpy as np
import pandas as pd 
import glob


# Remove all .log files in the current directory
for log_file in glob.glob("*.log"):
    try:
        os.remove(log_file)
        print(f"Removed log file: {log_file}")
    except OSError as e:
        print(f"Error removing file {log_file}: {e}")

from mishra03_optimization import Mishra03Optimization

    
    
ub = [10.0, 10.0]
lb = [-10.0, -10.0]
func = Mishra03Optimization()
NTrainingSamples = 50
RODEO_HOME = "/home/eoezkaya/RoDOP"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "mishra03.xml"


print("Number of samples used in the DoE = \n",NTrainingSamples)

ObjFunSamples = np.random.rand(NTrainingSamples,3)

for i in range(NTrainingSamples):
    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    ObjFunSamples[i,0] = x[0]
    ObjFunSamples[i,1] = x[1]
    ObjFunSamples[i,2] = func.evaluateFunction(x)	
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("mishra03.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



