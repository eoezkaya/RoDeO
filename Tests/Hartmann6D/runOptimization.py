import os
import numpy as np
import pandas as pd
import subprocess
import glob
# Remove all .log files in the current directory
for log_file in glob.glob("*.log"):
    try:
        os.remove(log_file)
        print(f"Removed log file: {log_file}")
    except OSError as e:
        print(f"Error removing file {log_file}: {e}")

compile_command = ["g++", "hartmann6D.cpp", "-o", "hartmann6D"]
try:
    subprocess.run(compile_command, check=True)
    print("Successfully compiled 'hartmann6D.cpp' to generate the 'hartmann6D' executable.")
except subprocess.CalledProcessError as e:
    print(f"Error during compilation of 'Xor.cpp': {e}")
    exit(1)  # Exit if compilation fails

# Define function to call "cola" executable and read result
def call_function_and_read_result(x):
    # Write the design variables to dv.dat
    with open("dv.dat", "w") as file:
        for item in x:
            file.write(f"{item}\n")
    try:
        subprocess.run(["./hartmann6D"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error calling 'Xor':", e)
        return None

    # Read the result from objFunVal.dat
    try:
        with open("objective.dat", "r") as file:
            result = float(file.readline().strip())
            print(f"Objective function value: {result}")
            return result
    except FileNotFoundError:
        print("Error: objective.dat file not found.")
        return None
    except ValueError:
        print("Error: Unable to convert result to float.")
        return None


dim = 6
# Define bounds and constants
ub = [1] * dim
lb = [0] * dim
NTrainingSamples = 100
RODEO_HOME = "/home/eoezkaya/RoDOP"
BIN_RODEO = os.path.join(RODEO_HOME, "build/rodeo")
configFilename = "hartmann6D.xml"

print(f"Number of samples used in the DoE: {NTrainingSamples}")

# Initialize array for design variables and objective function values
ObjFunSamples = np.zeros((NTrainingSamples, dim+1))

# Generate samples and evaluate function
for i in range(NTrainingSamples):
    x = np.random.rand(dim) * (np.array(ub) - np.array(lb)) + np.array(lb)
    ObjFunSamples[i, :dim] = x  # Store the design variables

    # Evaluate the objective function and store the result
    result = call_function_and_read_result(x)
    if result is not None:
        ObjFunSamples[i, dim] = result
    else:
        print(f"Warning: No result for sample {i}, setting objective value to NaN.")
        ObjFunSamples[i, dim] = np.nan

# Save the samples to a CSV file
df = pd.DataFrame(ObjFunSamples)
df.to_csv("hartmann6D.csv", header=False, index=False)
print("Results saved to 'hartmann6D.csv'.")

# Execute the external command with subprocess for better error handling
try:
    subprocess.run([BIN_RODEO, configFilename], check=True)
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Command '{BIN_RODEO} {configFilename}' failed with exit code {e.returncode}.")

