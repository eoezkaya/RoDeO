import os

BIN_RODEO = "../../../../build/rodeo"
configFilename = "himmelblau.cfg"
# compile the code that evaluates the objective function
os.system("g++ himmelblau.cpp -o himmelblau")

COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

