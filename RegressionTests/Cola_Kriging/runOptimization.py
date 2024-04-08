import os
import numpy as np
import pandas as pd 

import subprocess

def call_cola_and_read_result(x):
    with open("dv.dat", "w") as file:
        for item in x:
            file.write(str(item) + "\n")

    # Call the "cola" executable
    try:
        subprocess.run(["./cola"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None

    # Read the result from the file
    try:
        with open("objFunVal.dat", "r") as file:
            result = float(file.readline())
            print(result)
            return result
    except FileNotFoundError:
        print("Error: objFun.dat file not found.")
        return None
    except ValueError:
        print("Error: Unable to convert result to float.")
        return None



ub = 17*[4]
lb = 17*[0]

NTrainingSamples = 200
RODEO_HOME = "/home/eoezkaya/RoDeO"
BIN_RODEO = RODEO_HOME + "/build/rodeo"
configFilename = "cola.cfg"


print("Number of samples used in the DoE = \n",NTrainingSamples)

ObjFunSamples = np.random.rand(NTrainingSamples,18)

for i in range(NTrainingSamples):
    x = np.random.rand(17)
    for j in range(17):
        x[j] = x[j]*(ub[j]-lb[j])+lb[j]
    for j in range(17):
        ObjFunSamples[i,j] = x[j]    
    ObjFunSamples[i,17] = call_cola_and_read_result(x)
	


df = pd.DataFrame(ObjFunSamples)
df.to_csv("cola.csv",header=False, index=False)

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)



