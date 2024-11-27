import os
import numpy as np
import pandas as pd
import subprocess
import numpy as np 
import glob
from ackley_optimization import AckleyOptimization

# Remove all .log files in the current directory
for log_file in glob.glob("*.log"):
    try:
        os.remove(log_file)
        print(f"Removed log file: {log_file}")
    except OSError as e:
        print(f"Error removing file {log_file}: {e}")

func = AckleyOptimization()

# Set bounds and initialize function
ub = 20 * [21.768]
lb = 20 * [-30.768]


# Number of samples for Design of Experiment
NTrainingSamples = 50
RODEO_HOME = "/home/eoezkaya/RoDOP"
BIN_RODEO = os.path.join(RODEO_HOME, "build/rodeo")
configFilename = "ackley.xml"

print("Number of samples used in the DoE =", NTrainingSamples)

# Initialize array for design variables, function values, and gradients
ObjFunSamples = np.zeros((NTrainingSamples, 21))  


# Generate samples, evaluate function, and find the best sample
for i in range(NTrainingSamples):
    # Generate a random sample within bounds
    x = np.random.rand(20) * (np.array(ub) - np.array(lb)) + np.array(lb)
    ObjFunSamples[i, :20] = x  # Store design variables

    # Evaluate the function value
    function_value = func.evaluateFunction(x)
    ObjFunSamples[i, 20] = function_value  # Store function value

df = pd.DataFrame(ObjFunSamples)
df.to_csv("ackley.csv", header=False, index=False)
print("Results saved to 'ackley.csv'.")

# Execute the external command with subprocess for better error handling
try:
    subprocess.run([BIN_RODEO, configFilename], check=True)
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Command '{BIN_RODEO} {configFilename}' failed with exit code {e.returncode}.")

