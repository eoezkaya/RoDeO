import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from damavandi_optimization import DamavandiOptimization 
func = DamavandiOptimization()

df = pd.read_csv('optimizationHistory.csv')
dataMatrix = df.to_numpy()
x1 = dataMatrix[:,0]
x2 = dataMatrix[:,1]
f = dataMatrix[:,2]


fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot


xlist = np.linspace(0, 14, 100)
ylist = np.linspace(0, 14, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros([100,100])

for i in range(100):
    for j in range(100):
        x = [X[i,j], Y[i,j]]
        Z[i,j] = func.evaluateFunction(x)


plt.contour(X,Y,Z,50)
plt.scatter(x1, x2, c=f)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Damavandi function')

#plt.show()
plt.savefig("optimizationResults.png")


