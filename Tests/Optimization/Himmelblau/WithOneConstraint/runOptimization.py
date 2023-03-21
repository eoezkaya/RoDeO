import os

BIN_RODEO = "../../../../build/rodeo"
configFilename = "himmelblau.cfg"
# compile the code that evaluates the objective function
os.system("g++ himmelblau.cpp -o himmelblau")
# compile the code that evaluates the constraint
os.system("g++ constraint1.cpp -o constraint1")

os.system("cp himmelblauTrainingData.csv himmelblau.csv")
os.system("cp constraint1TrainingData.csv constraint1.csv")
COMMAND = BIN_RODEO + " " +  configFilename
os.system(COMMAND)

