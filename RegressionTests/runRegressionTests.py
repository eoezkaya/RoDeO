import os
import numpy as np
import pandas as pd 

import subprocess
from regressionTest import RegressionTest
import colorama
from colorama import Fore, Style


import xml.etree.ElementTree as ET

def read_objective_function(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Find the "ObjectiveFunction" element and extract its text content
    objective_function_element = root.find('ObjectiveFunction')
    
    if objective_function_element is not None:
        objective_function_value = float(objective_function_element.text)
        return objective_function_value
    else:
        # Handle the case where the "ObjectiveFunction" element is not found
        raise ValueError('ObjectiveFunction element not found in the XML file.')


def print_result(value, target):
    """
    Print "failed" in red if the value is less than the target, otherwise print "pass" in green.

    Parameters:
    - value: The value to compare.
    - target: The target value for comparison.
    """
    if value > target:
        print(f"{Fore.RED}Test failed{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Tess passed{Style.RESET_ALL}")

    print("\n")

def change_directory(new_path):
    try:
        os.chdir(new_path)
#        print(f"Changed directory to: {new_path}")
    except FileNotFoundError:
        print(f"Directory not found: {new_path}")
    except PermissionError:
        print(f"Permission denied to access directory: {new_path}")
        
        
        
import re



RODEO_HOME = "/home/eoezkaya/RoDeO"
DIR_REGRESSION_TEST = RODEO_HOME + "/RegressionTests"
print("################################# REGRESSION TESTS #################################")
print("RODEO_HOME :", RODEO_HOME)
print("DIRECTORY OF THE REGRESSION TESTS :", DIR_REGRESSION_TEST)
print("\n\n")
# Create an array (list) of RegressionTest instances
regression_tests = []


# Add instances to the array
regression_tests.append(RegressionTest("Gas Transmission Compressor Design","/Gas_transmission_compressor_design_Constrained_Kriging",
                                         1,    
                                         2971313.722654272, 
                                         3271313.722654272,   
                                         0.0001))
regression_tests.append(RegressionTest("Eggholder Unconstrained",
                                       "/Eggholder_Unconstrained_Kriging",  
                                       5,    
                                       -959.64, 
                                       -800,   
                                       0.01))
regression_tests.append(RegressionTest("Rosenbrock Constrained",
                                       "/Rosenbrock_Constrained_Kriging",
                                         1,    
                                         0.0, 
                                         1.0,   
                                         10.0))
regression_tests.append(RegressionTest("Wingweight Unconstrained",
                                       "/Wingweight_Unconstrained_Kriging",
                                         3,
                                         100.0,
                                         150.0,
                                         0.01))
regression_tests.append(RegressionTest("Himmelblau Unconstrained","/Himmelblau_Unconstrained_Kriging",  5,    0.0, 1.0,   100.0))
regression_tests.append(RegressionTest("Alpine02 5D Uncostrained","/Alpine02_5D_Unconstrained_Kriging", 3, -174.617, -30, 0.005))

#regression_tests.append(RegressionTest("/Himmelblau_Constrained_Kriging",20, 35.7))

# Accessing attributes of the instances in the array
total_mean_score = 0
testID = 1
for test in regression_tests:
    print("-" * 30)
    print("Test ID = ", testID)
    print("Problem name:", test.name)
    print("Number of Trials:", test.num_trials)
    print("Expected optimal Value:", test.optimal_value)
    
    
    DIR_TEST = DIR_REGRESSION_TEST + test.directory_name
    change_directory(DIR_TEST)
    script_path =  DIR_TEST + "/runOptimization.py"    
    
    objectiveFunctionValues = np.zeros(test.num_trials)
    for trial in range(test.num_trials):
        
        
        with open('RoDOpt.out', 'w') as output_file:
            subprocess.call(['python', script_path], stdout=output_file, stderr=subprocess.STDOUT)
        
        
        objective_function_value = read_objective_function("./globalOptimumDesign")

        if objective_function_value is not None:
            print(f"Objective Function Value: {objective_function_value}")
            objectiveFunctionValues[trial] = objective_function_value

    
    mean_value = np.mean(objectiveFunctionValues)
    print("Mean value = ", mean_value)
    mean_score = abs(mean_value - test.optimal_value)
    print("Difference = ", mean_score)
    mean_score = mean_score*test.scale_factor
    print("Mean score = ", mean_score)
    print("\n")
    print_result(mean_value, test.target_value)
    
    total_mean_score += mean_score 
    testID = testID+1


print("Total score for the mean = ", total_mean_score)

#RODEO_HOME = "/home/eoezkaya/RoDeO"
#BIN_RODEO = RODEO_HOME + "/build/rodeo"
#configFilename = "alpine02_5D.cfg"




#COMMAND = BIN_RODEO + " " +  configFilename
#os.system(COMMAND)





