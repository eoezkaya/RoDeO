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


# Compile "cola.cpp" to create the "cola" executable
compile_command = ["g++", "Cola.cpp", "-o", "cola"]
try:
    subprocess.run(compile_command, check=True)
    print("Successfully compiled 'cola.cpp' to generate the 'cola' executable.")
except subprocess.CalledProcessError as e:
    print(f"Error during compilation of 'cola.cpp': {e}")
    exit(1)  # Exit if compilation fails

# Define function to call "cola" executable and read result
def call_cola_and_read_result(x):
    # Write the design variables to dv.dat
    with open("dv.dat", "w") as file:
        for item in x:
            file.write(f"{item}\n")

    # Call the "cola" executable
    try:
        subprocess.run(["./cola"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error calling 'cola':", e)
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

# Define bounds and constants
ub = [4] * 17
lb = [0] * 17
NTrainingSamples = 200
RODEO_HOME = "/home/eoezkaya/RoDOP"
BIN_RODEO = os.path.join(RODEO_HOME, "build/rodeo")
configFilename = "cola.xml"

print(f"Number of samples used in the DoE: {NTrainingSamples}")

# Initialize array for design variables and objective function values
ObjFunSamples = np.zeros((NTrainingSamples, 18))

# Generate samples and evaluate function
for i in range(NTrainingSamples):
    x = np.random.rand(17) * (np.array(ub) - np.array(lb)) + np.array(lb)
    ObjFunSamples[i, :17] = x  # Store the design variables

    # Evaluate the objective function and store the result
    result = call_cola_and_read_result(x)
    if result is not None:
        ObjFunSamples[i, 17] = result
    else:
        print(f"Warning: No result for sample {i}, setting objective value to NaN.")
        ObjFunSamples[i, 17] = np.nan

# Save the samples to a CSV file
df = pd.DataFrame(ObjFunSamples)
df.to_csv("cola.csv", header=False, index=False)
print("Results saved to 'cola.csv'.")

# Execute the external command with subprocess for better error handling
try:
    subprocess.run([BIN_RODEO, configFilename], check=True)
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Command '{BIN_RODEO} {configFilename}' failed with exit code {e.returncode}.")

