import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('optimizationHistory.csv')
dataMatrix = df.to_numpy()
x1 = dataMatrix[:,0]
x2 = dataMatrix[:,1]
f = dataMatrix[:,2]


fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot




import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-1.5, 1.5, 100)
ylist = np.linspace(-0.5, 2.5, 100)


X, Y = np.meshgrid(xlist, ylist)
Z = (1-X)**2 + 100*(Y-X**2)**2


plt.contour(X,Y,Z,50)
plt.scatter(x1, x2, c=f)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Rosenbrock function')

#plt.show()
plt.savefig("optimizationResults.png")


