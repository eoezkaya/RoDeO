import os

BIN_RODEO = "../../../../build/rodeo"
configFilename = "himmelblau.cfg"
# compile the code that evaluates the objective function
os.system("g++ himmelblau.cpp -o himmelblau")
os.system("g++ himmelblauAdjointLowFi.cpp -o himmelblauLowFi")
os.system("cp himmelblauTrainingData.csv himmelblau.csv")
os.system("cp himmelblauLowFiTrainingData.csv himmelblauLowFi.csv")
COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

